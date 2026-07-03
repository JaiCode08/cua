from verifiers.base import VerificationResult


def test_verification_result_pass():
    result = VerificationResult.pass_({"key": "value"})
    assert result.passed is True
    assert result.reason is None
    assert result.details == {"key": "value"}


def test_verification_result_fail():
    result = VerificationResult.fail("Something went wrong", {"pid": 123})
    assert result.passed is False
    assert result.reason == "Something went wrong"
    assert result.details == {"pid": 123}
