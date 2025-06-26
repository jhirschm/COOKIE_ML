import torch
from torch.ao.quantization import get_default_qconfig

from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder
from ximg_to_ypdf_autoencoder import Zero_PulseClassifier
from denoising_util import *
# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '..', 'ml_backbone'))

# Add the utils directory to the Python path
sys.path.append(utils_dir)
from utils import DataMilking_Nonfat, DataMilking, DataMilking_SemiSkimmed, DataMilking_HalfAndHalf
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
    # datapath = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    # datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_06212024/"
    # datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06252024/"
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_07262024_0to1/test/"
    datapath_test = "/sdf/scratch/lcls/ds/prj/prjs2e21/scratch/fast_data_access/even-dist_Pulses_03302024/Processed_07262024_0to1/test/"
    datapath= datapath_test
    # datapath = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_06252024/"
    # dataset = DataMilking(root_dir=datapath, attributes=["energies", "phases", "npulses"], pulse_number=2)

    # datapath1 = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_06252024/"
    # datapath2 = "/sdf/data/lcls/ds/prj/prjs2e21/results/even-dist_Pulses_03302024/Processed_06252024/"
    # datapaths = [datapath1, datapath2]
    # pulse_specification = [{"pulse_number": 1, "pulse_number_max": None}, {"pulse_number": 0, "pulse_number_max": 10}]


    # data = DataMilking_Nonfat(root_dir=datapath, pulse_number=2, subset=4)
    # data = DataMilking_SemiSkimmed(root_dir=datapath, pulse_number=1, input_name="Ximg", labels=["Ypdf"])
    # data = DataMilking_HalfAndHalf(root_dirs=datapaths, pulse_handler = pulse_specification, input_name="Ximg", labels=["Ypdf"],transform=None, test_batch=None)
    data = DataMilking_HalfAndHalf(root_dirs=[datapath_test], pulse_handler = None, test_batch=1, input_name="Ximg", labels=["Ypdf"],transform=None)

    # data = DataMilking_SemiSkimmed(root_dir=datapath, pulse_number_max=10, input_name="Ximg", labels=["Ypdf"], test_batch=2)
    # Calculate the lengths for each split
    train_size = int(0 * len(data))
    val_size = int(0 * len(data))
    test_size = int(len(data) - train_size - val_size)

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])

     # Create data loaders
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)


    # Example usage
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])
   
    # decoder_layers = np.array([
    #     [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
    #     [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
    #     [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Tanh()]  # Example with Sigmoid activation
    #     # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    # ])
    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
        # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    ])
    
    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers)
    #     autoencoder,
    #     {nn.Linear,
    #      nn.Conv2d,
    #      nn.ConvTranspose2d},
    #      dtype=torch.qint8
    # )
    
    # Example usage
    conv_layers = [
        [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None],
        [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None]
    ]

    # Calculate the output size after conv layers
    def get_conv_output_size(input_size, conv_layers):
        x = torch.randn(input_size)
        model = nn.Sequential(*[layer for layer_pair in conv_layers for layer in layer_pair if layer is not None])
        x = model(x)
        return x.shape

    output_size = get_conv_output_size((1, 1, 512, 16), conv_layers)
    print(f"Output size after conv layers: {output_size}")

    # Use the calculated size for the fully connected layer input
    fc_layers = [
        [nn.Linear(output_size[1] * output_size[2] * output_size[3], 4), nn.ReLU()],
        [nn.Linear(4, 1), None]
    ]

    classifier = Zero_PulseClassifier(conv_layers, fc_layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    # model_save_dir = "/Users/jhirschm/Documents/MRCO/Data_Changed/Test"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07282024_multiPulse/outputs_fromEvenDist/"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_09032024_multiPulse_final/outputs_fromEvenDist/"
    best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06272024_singlePulse/testAutoencoder_best_model.pth"
    best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07282024_multiPulse/autoencoder_best_model.pth"
    best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_09022024_multiPulse_final/autoencoder_best_model.pth"
    best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_09032024_multiPulse_final/autoencoder_5_best_model.pth"
    # best_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_06302024_singlePulseAndZeroPulse_ErrorWeighted_3/autoencoder_best_model.pth"
    # best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07042024_zeroPredict/classifier_best_model.pth"
    best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_zeroPredict/classifier_best_model.pth"
    autoencoder.to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)

    classifier.to(device)
    state_dict = torch.load(best_model_zero_mask_path, map_location=device)
    print(state_dict.keys())
    # Remove keys related to side_network
    keys_to_remove = ['side_network.0.weight', 'side_network.0.bias']
    state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}


    classifier.load_state_dict(state_dict)
    

    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print("********* Non quantized Model *************")
    print(summary(autoencoder, input_size=(1, 1, 512, 16)))

    identifier = "testAutoencoder_quantized_eval"
    autoencoder_int8 = autoencoder.quantize(test_dataloader)
    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print("********* Quantized Model *************")
    print(summary(autoencoder_int8, input_size=(1, 1, 512, 16)))


    autoencoder_int8.evaluate_model(test_dataloader, criterion, device, save_results=True, results_dir=model_save_dir, results_filename=f"{identifier}_results.h5", zero_masking = True, zero_masking_model=classifier)
    results_file = os.path.join(model_save_dir, f"{identifier}_results.txt")
    with open(results_file, 'w') as f:
        f.write("Model Training Results\n")
        f.write("======================\n")
        f.write(f"Data Path: {datapath}\n")
        f.write(f"Model Save Directory: {model_save_dir}\n")
        f.write("\nModel Parameters and Hyperparameters\n")
        f.write("-----------------------------------\n")
        f.write("\nModel Architecture\n")
        f.write("------------------\n")
        f.write(f"Encoder Layers: {encoder_layers}\n")
        f.write(f"Decoder Layers: {decoder_layers}\n")
        f.write("\nAdditional Notes\n")
        f.write("----------------\n")
        f.write("Results for inspection on test. Running on even pulses but trained on 1 pulse. Max 10 pulses.\n")
        f.write((summary(autoencoder, input_size=(1, 1, 512, 16))))

    
    
if __name__ == "__main__":
    main()