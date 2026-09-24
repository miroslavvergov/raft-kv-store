"""RequestVote messages, built by keyword, shared by the consensus and node tests."""

from raftkv.consensus import RequestVoteRequest, RequestVoteResponse


def vote_request(*, term, candidate, last_log_term=0, last_log_index=0):
    """Return `candidate`'s RequestVote for `term`; its log defaults to empty."""
    return RequestVoteRequest(
        term=term,
        candidate_id=candidate,
        last_log_index=last_log_index,
        last_log_term=last_log_term,
    )


def granted(*, term):
    """Return an answer granting the vote, carrying `term`."""
    return RequestVoteResponse(term=term, vote_granted=True)


def refused(*, term):
    """Return an answer refusing the vote, carrying `term`."""
    return RequestVoteResponse(term=term, vote_granted=False)
