from regression_util import *
from pulse_phase_regression import RegressionModel



# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../..', 'ml_backbone'))
denoise_dir = os.path.abspath(os.path.join(current_dir, '../..', 'denoising'))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)


from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier
from utils import DataMilking_Nonfat, DataMilking, DataMilking_SemiSkimmed, DataMilking_HalfAndHalf, DataMilking_MilkCurds
from utils import CustomScheduler

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("MPS is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")
# device = torch.device("cpu")
def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Input Data Paths and Output Save Paths

    # Load Dataset and Feed to Dataloader
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/test/"

    pulse_specification = None


    data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=2)

