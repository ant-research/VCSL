from vcsl.metric import *


if __name__ == '__main__':
    segments = np.array([[0, 10], [2, 3], [4, 5]], dtype=np.float32)
    np.testing.assert_equal(seg_len(segments), 10.)

    segments = np.array([[2, 3], [0, 2], [2, 4], ], dtype=np.float32)
    np.testing.assert_equal(seg_len(segments), 4.)

    segments = np.array([[2, 3], [0, 1.5], [2.5, 4], ], dtype=np.float32)
    np.testing.assert_equal(seg_len(segments), 3.5)