from lexi_service.jobs.models import TERMINAL_JOB_STATUSES, JobStatus, may_transition


def test_only_queued_and_running_jobs_have_outgoing_transitions():
    assert may_transition(JobStatus.QUEUED, JobStatus.RUNNING)
    assert may_transition(JobStatus.RUNNING, JobStatus.SUCCEEDED)
    assert may_transition(JobStatus.RUNNING, JobStatus.QUEUED)
    assert not may_transition(JobStatus.SUCCEEDED, JobStatus.QUEUED)
    assert not may_transition(JobStatus.EXPIRED, JobStatus.SUCCEEDED)


def test_terminal_states_are_explicit_and_non_retriable():
    assert JobStatus.QUEUED not in TERMINAL_JOB_STATUSES
    assert JobStatus.RUNNING not in TERMINAL_JOB_STATUSES
    assert JobStatus.DEAD_LETTER in TERMINAL_JOB_STATUSES
