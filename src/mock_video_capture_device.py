from data_types import VideoCaptureDevice
import mock_data


class MockVideoCaptureDevice(VideoCaptureDevice):
    """ Mocks an external camera device. """

    def get_camera_image(self):
        """ Returns a random 256x256 numpy array. """
        return mock_data.get_mock_image(h=256, w=256)
