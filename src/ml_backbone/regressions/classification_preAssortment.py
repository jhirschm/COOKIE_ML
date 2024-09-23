from regression_util import *
from pulse_phase_regression import RegressionModel
from skimage.transform import iradon
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix, accuracy_score
import seaborn as sns
import math
import uuid  # To generate unique keys




# Get the directory of the currently running file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the utils directory relative to the current file's directory
utils_dir = os.path.abspath(os.path.join(current_dir, '../..', 'ml_backbone'))
classifier_dir = os.path.abspath(os.path.join(current_dir, '../../ml_backbone', 'classifiers'))

denoise_dir = os.path.abspath(os.path.join(current_dir, '../..', 'denoising'))

sys.path.append(utils_dir)
sys.path.append(denoise_dir)
sys.path.append(classifier_dir)
from lstm_pulseNum_classifier import CustomLSTMClassifier
from ximg_to_ypdf_autoencoder import Ximg_to_Ypdf_Autoencoder, Zero_PulseClassifier
from resnets import *
from resnets import BasicBlock, ResNet
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
    print("MPS/GPU is not available. Using CPU.")
import torch.nn.functional as F

# def remove_module_prefix(state_dict):
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith('module.'):
#                 new_state_dict[k[7:]] = v
#             else:
#                 new_state_dict[k] = v
#         return new_state_dict

# def nearest_power_of_2(n):
#     """Return the nearest power of 2 to the given number n."""
#     if n <= 0:
#         return 1  # In case n is 0 or negative, return the smallest power of 2
#     # Calculate the two closest powers of 2
#     lower_power = 2 ** math.floor(math.log2(n))
#     upper_power = 2 ** math.ceil(math.log2(n))
    
#     # Return the one that is closer to n
#     return lower_power if (n - lower_power) < (upper_power - n) else upper_power





# def save_filtered_data(model, dataloader, data_save_directory, file_prefix, max_examples_per_file, device, denoising=False,
#                        denoise_model=None, zero_mask_model=None, parallel=True):
    
#     # Check if the save directory exists, if not create it
#     if not os.path.exists(data_save_directory):
#         os.makedirs(data_save_directory)
#         print(f"Directory {data_save_directory} created.")

#     model.to(device)
#     if denoising and denoise_model is None and zero_mask_model is None:
#         raise ValueError("Denoising is enabled but no denoising model is provided")
#     if parallel:
#         model = torch.nn.DataParallel(model)
#         if denoising and denoise_model is not None and zero_mask_model is not None:
#             denoise_model = torch.nn.DataParallel(denoise_model)
#             zero_mask_model = torch.nn.DataParallel(zero_mask_model)
#             denoise_model.to(device)
#             zero_mask_model.to(device)
    
#     model.eval()  # Set the model to evaluation mode
    
#     example_count = 0
#     file_count = 0
#     save_file_path = None
#     save_h5f = None
    
#     def create_new_h5_file():
#         nonlocal save_file_path, save_h5f, file_count
#         if save_h5f is not None:
#             save_h5f.close()
#         save_filename = f"{file_prefix}_part{file_count}.h5"
#         save_file_path = os.path.join(data_save_directory, save_filename)
#         save_h5f = h5py.File(save_file_path, 'w')
#         file_count += 1
    
#     create_new_h5_file()  # Create the first file
    
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs, labels, phases, ypdfs, energies = batch
#             inputs, labels, phases, ypdfs, energies = inputs.to(device), labels.to(device), phases.to(device), ypdfs.to(device), energies.to(device)
            
#             input_preserved = inputs.clone().detach().cpu().numpy()
#             ypdf_preserved = ypdfs.clone().detach().cpu().numpy()
#             energy_preserved = energies.clone().detach().cpu().numpy()
#             phase_preserved = phases.clone().detach().cpu().numpy()
#             if denoising and denoise_model is not None and zero_mask_model is not None:
#                 inputs = torch.unsqueeze(inputs, 1)
#                 outputs = denoise_model(inputs)
#                 outputs = outputs.squeeze().to(device)
#                 if parallel:
#                     probs, zero_mask  = zero_mask_model.module.predict(inputs)
#                 else:
#                     probs, zero_mask  = zero_mask_model.predict(inputs)
#                 zero_mask = torch.unsqueeze(zero_mask, 2).to(device, torch.float32)
#                 outputs = outputs * zero_mask
#                 inputs = outputs.detach().clone().to(device, torch.float32)

#             else:
#                 inputs = inputs.to(device, torch.float32)
            
#             if parallel:
#                 probs, preds = model.module.predict(inputs)
#             else:
#                 probs, preds = model.predict(inputs)
#             predicted_pulse_single_label = np.argmax(probs.cpu().numpy(), axis=1)
            
#             # Filtering only the examples classified as 2 pulses
#             two_pulse_indices = np.where(predicted_pulse_single_label == 2)[0]
            
#             if len(two_pulse_indices) > 0:
#                 # Save these examples to the current h5 file
#                 for idx in two_pulse_indices:
#                     # Generate a unique image key using file_count and example_count
#                     image_key = f"img_{file_count}_{example_count}_{str(uuid.uuid4())[:8]}"
                    
#                     # Save the filtered data in the HDF5 file under the new image_key
#                     grp = save_h5f.create_group(image_key)
#                     grp.create_dataset("Ximg", data=input_preserved[idx])
#                     grp.create_dataset("Ypdf", data=ypdf_preserved[idx])
#                     grp.attrs["energies"] = energy_preserved[idx]
#                     grp.attrs["phases"] = phase_preserved[idx]
#                     grp.attrs["npulses"] = 2  # Set as 2 pulses
                    
#                     example_count += 1
                    
#                     # Check if we need to create a new file
#                     if example_count % max_examples_per_file == 0:
#                         create_new_h5_file()
    
#     if save_h5f is not None:
#         save_h5f.close()  # Close the last h5 file

# def main():
    
#     dtype = torch.float32

#     # Load Dataset and Feed to Dataloader
#     datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/test/"
#     datapath_train = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/train/"
#     datapath_save_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/test/assorted/"
#     datapath_save_train = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/train/assorted/"

#     data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=2, ypdfs_included=True, energies_included=True, energies_included_max=2)
#     data_train = DataMilking_MilkCurds(root_dirs=[datapath_train], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=2, ypdfs_included=True,  energies_included=True, energies_included_max=2)


   

#     test_dataset = data_test
#     train_dataset = data_train
    
   
#     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)


#     model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/preSortingDoublePulse_09222024/"
#     # model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_09022024_Resnext34_dif_Ximg_Denoised_1/evaluate_outputs/"

#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir)
    
#     # identifier = "Resnext34_2hotsplit_EMDloss_Ypdf_train"
#     identifier = "LSTM_For_PreSorting_DoublePulse"
    

#     '''
#     denoising
#     '''
#     best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07282024_multiPulse/autoencoder_best_model.pth"
#     best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_zeroPredict/classifier_best_model.pth"
    
#     # Example usage
#     encoder_layers = np.array([
#         [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
#         [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
#         [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])
   
#     decoder_layers = np.array([
#         [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
#         [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
#         [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
#         # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
#     ])

#     # Example usage
#     conv_layers = [
#         [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()],
#         [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None],
#         [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()],
#         [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None]
#     ]

#     output_size = get_conv_output_size((1, 1, 512, 16), conv_layers)

#     # Use the calculated size for the fully connected layer input
#     fc_layers = [
#         [nn.Linear(output_size[1] * output_size[2] * output_size[3], 4), nn.ReLU()],
#         [nn.Linear(4, 1), None]
#     ]
#     zero_model = Zero_PulseClassifier(conv_layers, fc_layers)
#     zero_model.to(device)
#     state_dict = torch.load(best_model_zero_mask_path, map_location=device)
#     keys_to_remove = ['side_network.0.weight', 'side_network.0.bias']
#     state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
#     zero_model.load_state_dict(state_dict)

#     autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers, outputEncoder=False)
#     autoencoder.to(device)
#     state_dict = torch.load(best_autoencoder_model_path, map_location=device)
#     autoencoder.load_state_dict(state_dict)



#     data = {
#         "hidden_size": 128,#128
#         "num_lstm_layers": 3,
#         "bidirectional": True,
#         "fc_layers": [32, 64],
#         "dropout": 0.2,
#         "lstm_dropout": 0.2,
#         "layerNorm": False,
#         # Other parameters are default or not provided in the example
#     }   

#     # Assuming input_size and num_classes are defined elsewhere
#     input_size = 512  # Define your input size
#     num_lstm_classes = 5   # Example number of classes

#     # Instantiate the CustomLSTMClassifier
#     classModel = CustomLSTMClassifier(
#         input_size=input_size,
#         hidden_size=data['hidden_size'],
#         num_lstm_layers=data['num_lstm_layers'],
#         num_classes=num_lstm_classes,
#         bidirectional=data['bidirectional'],
#         fc_layers=data['fc_layers'],
#         dropout_p=data['dropout'],
#         lstm_dropout=data['lstm_dropout'],
#         layer_norm=data['layerNorm'],
#         ignore_output_layer=False  # Set as needed based on your application
#     )
#     classModel.to(device)

#     best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_09052024_5classCase/testLSTM_XimgDenoised_best_model.pth"
#     state_dict = torch.load(best_mode_classifier, map_location=device)

#     state_dict = remove_module_prefix(state_dict)
#     # for key in state_dict.keys():
#     #     print(key, state_dict[key].shape)
#     classModel.load_state_dict(state_dict)

   
#     # Calculate the lengths for each split

#    # Get the number of .h5 files in the train directory
#     num_train_files = len([f for f in os.listdir(datapath_train) if f.endswith('.h5') and os.path.isfile(os.path.join(datapath_train, f))])
#     print(f"Number of .h5 files in the train directory: {num_train_files}")

#     # Get the number of .h5 files in the test directory
#     num_test_files = len([f for f in os.listdir(datapath_test) if f.endswith('.h5') and os.path.isfile(os.path.join(datapath_test, f))])
#     print(f"Number of .h5 files in the test directory: {num_test_files}")
    
#     train_size = len(data_train)
#     print(f"Train size: {train_size}")
#     test_size = len(data_test)
#     print(f"Test size: {test_size}")
    
#     test_max_examples_per_file = test_size // num_test_files
#     rounded_test_max_examples_per_file = nearest_power_of_2(test_max_examples_per_file)
#     print(f"Rounded test max examples per file: {rounded_test_max_examples_per_file}")

#     train_max_examples_per_file = train_size // num_train_files
#     rounded_train_max_examples_per_file = nearest_power_of_2(train_max_examples_per_file)
#     print(f"Rounded train max examples per file: {rounded_train_max_examples_per_file}")

#     save_filtered_data(classModel, test_dataloader, data_save_directory = datapath_save_test, file_prefix = "preclassified2", max_examples_per_file = rounded_test_max_examples_per_file, device = device, denoising=True, denoise_model =autoencoder,
#                 zero_mask_model = zero_model, parallel=True)
    
#     save_filtered_data(classModel, train_dataloader, data_save_directory = datapath_save_train, file_prefix = "preclassified2", max_examples_per_file = rounded_train_max_examples_per_file, device = device, denoising=True, denoise_model =autoencoder,
#                 zero_mask_model = zero_model, parallel=True)
    
# if __name__ == "__main__":
#     main()