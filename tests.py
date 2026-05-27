import warnings
import unittest
from unittest.mock import MagicMock, patch

warnings.filterwarnings(
    "ignore",
    message=r"subprocess .* is still running",
    category=ResourceWarning
)

warnings.filterwarnings(
    "ignore",
    message=r"unclosed file .*ActiveX\.bat",
    category=ResourceWarning
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="ansys.motorcad.core.*"
)

from create_first_data import MCAD
import utils

class MyTestCase(unittest.TestCase):

    def test_mcad_initialization_starts_motorcad_and_eduvpn(self):
        with MCAD() as mcad:
            self.assertIsInstance(mcad, MCAD)
            self.assertIsNotNone(mcad.mcad)
            self.assertTrue(utils.is_running("eduVPN.Client.exe"))
            self.assertIsNotNone(mcad.eduVPN)
            self.assertIsNone(mcad.eduVPN.poll())

    @patch("create_first_data.utils.stop_process_by_name")
    @patch("create_first_data.utils.stop_process")
    @patch("create_first_data.utils.wait_until_running")
    @patch("create_first_data.utils.start_process")
    @patch("create_first_data.pymotorcad.MotorCAD")
    def test_open_motorcad_file_raises_error_when_file_cannot_be_opened(
            self,
            mock_motorcad_class,
            mock_start_process,
            mock_wait_until_running,
            mock_stop_process,
            mock_stop_process_by_name
    ):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_start_process.return_value = mock_process

        mock_motorcad = MagicMock()
        mock_motorcad_class.return_value = mock_motorcad

        with MCAD() as mcad:
            with self.assertRaises(FileNotFoundError):
                mcad.open_motorcad_file("missing_file.mot")


if __name__ == '__main__':
    unittest.main()