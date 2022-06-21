import pytest
import subprocess


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.4, 1, 10])
@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5])
def test_params(alpha, ratio):
    subprocess.call(['/home/lichenglong/anaconda3/envs/fedme/bin/python3', 'untitled.py', '--model=cnn',
                     '--sampling_ratio={}'.format(ratio), '--alpha={}'.format(alpha), '--batch_size=32',
                     '--comm_round=200'])


if __name__ == "__main__":
    pytest.main(["-s", "param_test.py", '-n', '3'])
