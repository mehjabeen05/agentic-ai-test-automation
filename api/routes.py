"""API routes: orchestrates the existing agents/generators/executors/repositories.

No business logic lives here — every route is a thin translation between
an HTTP request/response and a call into an existing module (agents/,
generators/, executor/, core/repositories.py). Nothing here re-implements
LLM calls, code generation, execution, or SQL; see api/dependencies.py for
how each component is constructed and injected.

Security posture for this layer specifically:
    - Every "run" or "generate" request identifies its target only by an
      opaque id (`test_case_id`, `requirement_id`, `execution_id`) — never
      a filesystem path, shell command, or code. `/tests/run` resolves a
      `test_case_id` to a path using the SAME deterministic function the
      generator itself uses (`build_generated_filename`), and then hands
      that path to `TestRunner`, which independently re-validates it (path
      containment, extension, AST safety) regardless of how it was built.
    - Nothing returned by the LLM is ever executed here or anywhere it
      calls into.
"""

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from agents.failure_analysis_agent import FailureAnalysisAgent, FailureAnalysisError
from agents.requirement_agent import RequirementAgent, RequirementAgentError
from agents.test_generation_agent import TestCaseGenerationError, TestCaseGeneratorAgent
from api.dependencies import (
    get_execution_repository,
    get_failure_analysis_agent,
    get_failure_analysis_repository,
    get_healing_executor,
    get_healing_repository,
    get_page_context,
    get_playwright_code_generator,
    get_requirement_agent,
    get_requirement_repository,
    get_test_case_generator_agent,
    get_test_case_repository,
    get_test_runner,
)
from api.schemas import (
    ExecutionSummary,
    FailureAnalysisRequest,
    FailureAnalysisResponse,
    HealingRequest,
    HealingResponse,
    RequirementRequest,
    RequirementResponse,
    RequirementSummary,
    StatsResponse,
    TestCaseGenerationRequest,
    TestCaseGenerationResponse,
    TestCaseSummary,
    TestExecutionRequest,
    TestExecutionResponse,
    TestGenerationRequest,
    TestGenerationResponse,
    ValidationSummary,
)
from core.config import get_settings
from core.logger import get_logger
from core.models import ExecutionStatus, FailureAnalysis, FailureAnalysisInput, TestRequirement
from core.repositories import (
    ExecutionRepository,
    FailureAnalysisRepository,
    HealingRepository,
    RequirementRepository,
    TestCaseRepository,
)
from executor.healing_executor import HealingExecutor
from executor.test_runner import TestRunner
from generators.code_validator import CodeValidationError
from generators.playwright_generator import (
    PlaywrightCodeGenerator,
    PlaywrightGenerationError,
    build_generated_filename,
)

logger = get_logger(__name__)

router = APIRouter()

# Matches Playwright's own error text, e.g.:
#   waiting for locator("#username")
# Used only to bridge already-stored execution data into a selector string
# for healing — this is request-orchestration glue, not a re-implementation
# of any agent's logic.
_LOCATOR_PATTERN = re.compile(r"""locator\((["'])(.*?)\1\)""")


def _extract_selector_from_error(error_text: str | None) -> str | None:
    if not error_text:
        return None
    match = _LOCATOR_PATTERN.search(error_text)
    return match.group(2) if match else None


# --- Stats (dashboard summary) ---


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["stats"],
    summary="Dashboard summary counters",
    description="Aggregates existing repository data — no new business logic, just counting.",
)
def get_stats(
    test_case_repository: TestCaseRepository = Depends(get_test_case_repository),
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
) -> StatsResponse:
    total_test_cases = len(test_case_repository.get_all())
    executions = execution_repository.get_all()

    passed = sum(1 for e in executions if e.status == ExecutionStatus.PASSED)
    failed = sum(1 for e in executions if e.status in (ExecutionStatus.FAILED, ExecutionStatus.ERROR))
    healed = sum(1 for e in executions if e.healed)
    success_rate = (passed / len(executions) * 100) if executions else 0.0

    return StatsResponse(
        total_test_cases=total_test_cases,
        passed=passed,
        failed=failed,
        healed=healed,
        success_rate=round(success_rate, 1),
    )


# --- Requirements ---


@router.post(
    "/requirements",
    response_model=RequirementResponse,
    status_code=201,
    tags=["requirements"],
    summary="Analyze a natural-language requirement",
    description="Runs the Requirement Agent (Step 4) and persists the result.",
)
def create_requirement(
    request: RequirementRequest,
    agent: RequirementAgent = Depends(get_requirement_agent),
) -> RequirementResponse:
    try:
        requirement = TestRequirement(text=request.requirement)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid requirement text.") from exc

    try:
        analysis = agent.analyze(requirement)
    except RequirementAgentError as exc:
        logger.error("Requirement analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to analyze the requirement.") from exc

    if agent.last_requirement_id is None:
        logger.error("Requirement analyzed successfully but could not be persisted.")
        raise HTTPException(
            status_code=500, detail="Requirement analysis succeeded but could not be saved."
        )

    return RequirementResponse(requirement_id=agent.last_requirement_id, analysis=analysis)


@router.get(
    "/requirements",
    response_model=list[RequirementSummary],
    tags=["requirements"],
    summary="List every stored requirement",
)
def list_requirements(
    repository: RequirementRepository = Depends(get_requirement_repository),
) -> list[RequirementSummary]:
    records = repository.get_all()
    return [
        RequirementSummary(id=r.id, requirement_text=r.requirement_text, created_at=r.created_at)
        for r in records
    ]


@router.get(
    "/requirements/{requirement_id}/test-cases",
    response_model=list[TestCaseSummary],
    tags=["requirements"],
    summary="List the test cases generated for a requirement",
)
def get_test_cases_for_requirement(
    requirement_id: int,
    requirement_repository: RequirementRepository = Depends(get_requirement_repository),
    test_case_repository: TestCaseRepository = Depends(get_test_case_repository),
) -> list[TestCaseSummary]:
    if requirement_repository.get_by_id(requirement_id) is None:
        raise HTTPException(status_code=404, detail="Requirement not found.")

    records = test_case_repository.get_by_requirement(requirement_id)
    return [
        TestCaseSummary(
            id=r.id,
            requirement_id=r.requirement_id,
            test_case_id=r.test_case_id,
            title=r.title,
            description=r.description,
            priority=r.priority,
            type=r.type,
            created_at=r.created_at,
        )
        for r in records
    ]


# --- Test case generation ---


@router.post(
    "/test-cases",
    response_model=TestCaseGenerationResponse,
    status_code=201,
    tags=["test-cases"],
    summary="Generate test cases for a previously analyzed requirement",
    description="Runs the Test Case Generator Agent (Step 5) and persists each test case.",
)
def generate_test_cases(
    request: TestCaseGenerationRequest,
    agent: TestCaseGeneratorAgent = Depends(get_test_case_generator_agent),
    requirement_repository: RequirementRepository = Depends(get_requirement_repository),
) -> TestCaseGenerationResponse:
    requirement_record = requirement_repository.get_by_id(request.requirement_id)
    if requirement_record is None:
        raise HTTPException(status_code=404, detail="Requirement not found.")

    if requirement_record.analysis is None:
        raise HTTPException(
            status_code=400,
            detail="This requirement has no stored analysis to generate test cases from.",
        )

    try:
        test_cases = agent.generate(requirement_record.analysis, requirement_id=request.requirement_id)
    except TestCaseGenerationError as exc:
        logger.error("Test case generation failed for requirement %d: %s", request.requirement_id, exc)
        raise HTTPException(status_code=500, detail="Failed to generate test cases.") from exc

    return TestCaseGenerationResponse(requirement_id=request.requirement_id, test_cases=test_cases)


# --- Playwright code generation ---


@router.post(
    "/tests/generate",
    response_model=TestGenerationResponse,
    status_code=201,
    tags=["tests"],
    summary="Generate a Playwright test from a stored test case",
    description="Runs the Playwright Code Generator and AST code validator (Step 6).",
)
def generate_playwright_test(
    request: TestGenerationRequest,
    generator: PlaywrightCodeGenerator = Depends(get_playwright_code_generator),
    test_case_repository: TestCaseRepository = Depends(get_test_case_repository),
) -> TestGenerationResponse:
    test_case = test_case_repository.get_full_test_case(request.test_case_id)
    if test_case is None:
        raise HTTPException(status_code=404, detail="Test case not found.")

    try:
        code = generator.generate_code(test_case)
        saved_path = generator.save_code(test_case, code)
    except CodeValidationError as exc:
        logger.error("Generated code failed validation for '%s': %s", request.test_case_id, exc)
        raise HTTPException(
            status_code=400, detail="Generated code failed safety validation."
        ) from exc
    except PlaywrightGenerationError as exc:
        logger.error("Playwright generation failed for '%s': %s", request.test_case_id, exc)
        raise HTTPException(status_code=500, detail="Failed to generate the Playwright test.") from exc

    # Built from the configured workspace *string* (e.g. "tests/generated"),
    # not saved_path.relative_to(some_root) — the workspace's absolute
    # location can vary (e.g. under a tmp dir in tests), but saved_path is
    # always a direct child of it, so this is robust either way.
    generated_file = f"{get_settings().allowed_workspace_dir}/{saved_path.name}"
    return TestGenerationResponse(
        test_case_id=request.test_case_id,
        generated_file=generated_file,
        validation=ValidationSummary(valid=True),
    )


# --- Test execution ---


@router.post(
    "/tests/run",
    response_model=TestExecutionResponse,
    tags=["tests"],
    summary="Run a generated test by test_case_id",
    description=(
        "Resolves test_case_id to its generated file deterministically and "
        "runs it through the safe TestRunner (Step 7). Never accepts a "
        "filesystem path directly."
    ),
)
def run_generated_test(
    request: TestExecutionRequest,
    runner: TestRunner = Depends(get_test_runner),
    test_case_repository: TestCaseRepository = Depends(get_test_case_repository),
) -> TestExecutionResponse:
    if test_case_repository.get_by_test_case_id(request.test_case_id) is None:
        raise HTTPException(status_code=404, detail="Test case not found.")

    filename = build_generated_filename(request.test_case_id)
    candidate_path = get_settings().allowed_workspace_path / filename

    # TestRunner independently re-validates this path (containment,
    # extension, AST safety) — it is never trusted just because we built it.
    result = runner.run_test(candidate_path)

    return TestExecutionResponse(
        execution_id=runner.last_execution_id,
        test_case_id=result.test_id,
        status=result.status,
        duration=result.duration,
        error=result.error,
        screenshot=result.screenshot,
        healed=result.healed,
    )


@router.get(
    "/test-cases/{test_case_id}/executions",
    response_model=list[ExecutionSummary],
    tags=["tests"],
    summary="Execution history for a test case",
)
def get_execution_history(
    test_case_id: str,
    test_case_repository: TestCaseRepository = Depends(get_test_case_repository),
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
) -> list[ExecutionSummary]:
    if test_case_repository.get_by_test_case_id(test_case_id) is None:
        raise HTTPException(status_code=404, detail="Test case not found.")

    # Executions are recorded under the runtime id TestRunner derives from
    # the generated filename (e.g. "test_tc_login_001"), not the natural
    # test_case_id (e.g. "TC_LOGIN_001") — see core/repositories.py's
    # module docstring. Convert using the same public helper the generator
    # itself uses, so this never duplicates that sanitization rule.
    runtime_test_id = build_generated_filename(test_case_id).removesuffix(".py")
    records = execution_repository.get_history(runtime_test_id)
    return [_execution_record_to_summary(r) for r in records]


@router.get(
    "/executions/{execution_id}",
    response_model=ExecutionSummary,
    tags=["tests"],
    summary="A single execution by id",
)
def get_execution(
    execution_id: int,
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
) -> ExecutionSummary:
    record = execution_repository.get_by_id(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Execution not found.")
    return _execution_record_to_summary(record)


def _execution_record_to_summary(record) -> ExecutionSummary:
    return ExecutionSummary(
        id=record.id,
        test_case_id=record.test_case_id,
        status=record.status,
        duration=record.duration,
        error=record.error,
        screenshot=record.screenshot,
        healed=record.healed,
        created_at=record.created_at,
    )


# --- Failure analysis ---


@router.post(
    "/tests/analyze-failure",
    response_model=FailureAnalysisResponse,
    status_code=201,
    tags=["tests"],
    summary="Analyze a failed execution",
    description="Runs the Failure Analysis Agent (Step 8) against a stored execution.",
)
def analyze_failure(
    request: FailureAnalysisRequest,
    agent: FailureAnalysisAgent = Depends(get_failure_analysis_agent),
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
) -> FailureAnalysisResponse:
    execution = execution_repository.get_by_id(request.execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found.")

    if execution.status in (ExecutionStatus.PASSED, ExecutionStatus.SKIPPED):
        raise HTTPException(
            status_code=400, detail="This execution did not fail; there is nothing to analyze."
        )

    analysis_input = FailureAnalysisInput(
        test_id=execution.test_case_id,
        status=execution.status,
        duration=execution.duration,
        stdout=execution.stdout or "",
        stderr=execution.stderr or "",
        error=execution.error,
        screenshot_path=execution.screenshot,
    )

    try:
        analysis = agent.analyze(analysis_input, execution_id=request.execution_id)
    except FailureAnalysisError as exc:
        logger.error("Failure analysis failed for execution %d: %s", request.execution_id, exc)
        raise HTTPException(status_code=500, detail="Failed to analyze the failure.") from exc

    return FailureAnalysisResponse(execution_id=request.execution_id, **analysis.model_dump())


@router.get(
    "/executions/{execution_id}/failure-analysis",
    response_model=list[FailureAnalysisResponse],
    tags=["tests"],
    summary="Stored failure analyses for an execution",
)
def get_failure_analysis(
    execution_id: int,
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
    failure_analysis_repository: FailureAnalysisRepository = Depends(get_failure_analysis_repository),
) -> list[FailureAnalysisResponse]:
    if execution_repository.get_by_id(execution_id) is None:
        raise HTTPException(status_code=404, detail="Execution not found.")

    records = failure_analysis_repository.get_for_execution(execution_id)
    return [
        FailureAnalysisResponse(
            execution_id=execution_id,
            failure_type=r.failure_type,
            summary=r.summary,
            root_cause=r.root_cause,
            suggested_fix=r.suggested_fix,
            confidence=r.confidence,
            is_likely_environment_issue=r.is_likely_environment_issue,
            is_likely_test_issue=r.is_likely_test_issue,
        )
        for r in records
    ]


# --- Healing ---


@router.post(
    "/tests/heal",
    response_model=HealingResponse,
    status_code=201,
    tags=["tests"],
    summary="Attempt controlled selector healing for a failed execution",
    description=(
        "Runs the Healing Agent/Executor (Step 9). Requires a prior failure "
        "analysis on this execution. Only ever proposes/validates SELECTOR "
        "STRINGS — never executes LLM-generated code."
    ),
)
def heal_selector(
    request: HealingRequest,
    healing_executor: HealingExecutor = Depends(get_healing_executor),
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
    failure_analysis_repository: FailureAnalysisRepository = Depends(get_failure_analysis_repository),
    page_context=Depends(get_page_context),
) -> HealingResponse:
    execution = execution_repository.get_by_id(request.execution_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Execution not found.")

    failure_records = failure_analysis_repository.get_for_execution(request.execution_id)
    if not failure_records:
        raise HTTPException(
            status_code=400,
            detail="No failure analysis exists for this execution yet; analyze the failure first.",
        )
    latest_failure = failure_records[-1]
    failure_analysis = FailureAnalysis(
        failure_type=latest_failure.failure_type,
        summary=latest_failure.summary,
        root_cause=latest_failure.root_cause,
        suggested_fix=latest_failure.suggested_fix,
        confidence=latest_failure.confidence,
        is_likely_environment_issue=latest_failure.is_likely_environment_issue,
        is_likely_test_issue=latest_failure.is_likely_test_issue,
    )

    original_selector = _extract_selector_from_error(execution.error)
    if original_selector is None:
        raise HTTPException(
            status_code=400,
            detail="Could not determine the failed selector from the stored execution data.",
        )

    with page_context(request.url) as page:
        result = healing_executor.attempt_heal(
            test_id=execution.test_case_id,
            page=page,
            original_selector=original_selector,
            failed_action="click",
            retry_action=lambda p, selector: p.click(selector),
            failure=failure_analysis,
            execution_id=request.execution_id,
        )

    return HealingResponse(
        execution_id=request.execution_id,
        status=result.status,
        original_selector=result.original_selector,
        candidate_selectors=result.candidate_selectors,
        selected_selector=result.selected_selector,
        retry_succeeded=result.retry_succeeded,
        confidence=result.confidence,
        reason=result.reason,
    )


@router.get(
    "/executions/{execution_id}/healing",
    response_model=list[HealingResponse],
    tags=["tests"],
    summary="Stored healing attempts for an execution",
)
def get_healing_history(
    execution_id: int,
    execution_repository: ExecutionRepository = Depends(get_execution_repository),
    healing_repository: HealingRepository = Depends(get_healing_repository),
) -> list[HealingResponse]:
    if execution_repository.get_by_id(execution_id) is None:
        raise HTTPException(status_code=404, detail="Execution not found.")

    records = healing_repository.get_for_execution(execution_id)
    return [
        HealingResponse(
            execution_id=execution_id,
            status=r.status,
            original_selector=r.original_selector,
            candidate_selectors=r.candidate_selectors,
            selected_selector=r.selected_selector,
            retry_succeeded=r.retry_succeeded,
            confidence=r.confidence,
            reason=r.reason,
        )
        for r in records
    ]
