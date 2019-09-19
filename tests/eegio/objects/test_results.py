import pytest

from eegio.base.objects.dataset.result_object import Result


class TestResult:
    @pytest.mark.usefixture("result")
    def test_result_errors(self, result: Result):
        """
        Test error and warnings raised by Result class.

        :param result: (Result)
        :return: None
        """
        pass

    @pytest.mark.usefixture("result")
    def test_result(self, result: Result):
        """
        Test code runs without errors through all functions with dummy data.

        :param result:
        :return:
        """
        pass
