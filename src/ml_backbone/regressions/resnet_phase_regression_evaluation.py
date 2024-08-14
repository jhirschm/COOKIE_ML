from regression_util import *
from pulse_phase_regression import RegressionModel



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
    print("MPS is not available. Using CPU.")
import torch.nn.functional as F

def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

def get_phase(outputs, num_classes, max_val=2*torch.pi):
    # Convert the model outputs to probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)
    
    # Calculate a weighted sum of the class indices
    indices = torch.arange(num_classes, device=outputs.device, dtype=outputs.dtype)
    phase_values = torch.sum(probabilities * indices, dim=1)

    # Map the weighted sum to a phase value between 0 and 2*pi
    phase_values = phase_values * (max_val / num_classes)
    
    # Add an extra dimension to match expected output shape
    phase_values = torch.unsqueeze(phase_values, 1)
    phase_values = phase_values.to(torch.float32)
    return phase_values
    


def test_model(model, test_dataloader,  model_save_dir, identifier, device, criterion,
                    denoising=False, denoise_model=None, zero_mask_model=None, parallel=True, num_classes=1000):

    test_losses = []
    true_phase_list = []
    predicted_phase_list = []
    running_test_loss = 0
    model.to(device)

    if denoising and denoise_model is None and zero_mask_model is None:
        raise ValueError("Denoising is enabled but no denoising model is provided")
    model.eval()  # Set the model to evaluation mode, ensures no dropout is applied
    # Iterate through the test data
        
    with torch.no_grad():
        for batch in test_dataloader:

            inputs, labels, phases = batch
            inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                
            if denoising and denoise_model is not None and zero_mask_model is not None:
                
                denoise_model.eval()
                zero_mask_model.eval()
                
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device, torch.float32)
                # labels = labels[0]
                
                outputs = denoise_model(inputs)
                outputs = outputs.squeeze()
                outputs = outputs.to(device)
                if parallel:
                    probs, zero_mask  = zero_mask_model.module.predict(inputs)
                else:
                    probs, zero_mask  = zero_mask_model.predict(inputs)
                zero_mask = zero_mask.to(device)
                # zero mask either 0 or 1
                # change size of zero mask to match the size of the output dimensions so can broadcast in multiply
                zero_mask = torch.unsqueeze(zero_mask,2)
                zero_mask = zero_mask.to(device, torch.float32)

                outputs = outputs * zero_mask
                inputs = outputs.to(device, torch.float32)

            else: 
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device, torch.float32)
                        
                    
                    
            outputs = model(inputs).to(device)
            outputs = get_phase(outputs, num_classes, max_val=2*torch.pi)
            phases = phases.to(torch.float32)
            loss = criterion(outputs, phases)
                    
            running_test_loss += loss.item()
            
            test_loss = running_test_loss / len(test_dataloader)
            test_losses.append(test_loss)

            predicted_phases = outputs.cpu().numpy()
            true_phases = phases.cpu().numpy()
            for output, true_phase in zip(predicted_phases, true_phases):
                    predicted_phase_list.append(output.item())  
                    true_phase_list.append(true_phase.item())  

            predicted_phase_array = np.array(predicted_phase_list)
            true_phase_array = np.array(true_phase_list)
            print(f"Predicted Phase Differences: {predicted_phase_array[:10]}")
            print(f"True Phase Differences: {true_phase_array[:10]}")
              # Calculate the mean squared error                      
            
                        


        
        plot_path = os.path.join(model_save_dir, identifier + "_TruePred.pdf")
        # Plot the values
        plt.figure(figsize=(10, 6))
        plt.scatter(true_phase_array, predicted_phase_array, color='blue', label='Predicted vs True')
        plt.plot([true_phase_array.min(), true_phase_array.max()], 
                [true_phase_array.min(), true_phase_array.max()], 
                color='red', linestyle='--', label='Ideal Prediction')
        plt.xlabel('True Phase Differences')
        plt.ylabel('Predicted Phase Differences')
        plt.title('True vs Predicted Phase Differences')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(plot_path)

        plot_path = os.path.join(model_save_dir, identifier + "_SinTruePred.pdf")
        # Plot the values
        plt.figure(figsize=(10, 6))
        plt.scatter(np.sin(true_phase_array), np.sin(predicted_phase_array), color='blue', label='Predicted vs True')
        # plt.plot([true_phase_differences_array.min(), true_phase_differences_array.max()], 
        #         [true_phase_differences_array.min(), true_phase_differences_array.max()], 
        #         color='red', linestyle='--', label='Ideal Prediction')
        plt.xlabel('True Phase Differences')
        plt.ylabel('Predicted Phase Differences')
        plt.title('True vs Predicted Phase Differences')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(plot_path)

        return 1
                










                

def main():
    
    dtype = torch.float32


   
    fake_input = torch.randn(1, 1, 512, 16, device=device, dtype=dtype)
    
    # model = ResNet(block=BasicBlock, layers=[2,2,1,1], num_classes=1000)
    num_classes = 2000
    # model = resnet152(num_classes=num_classes)
    model = resnet18(num_classes=num_classes)

    model = model.to(device).to(dtype)


    try:
        output = model(fake_input)
        print("Output shape:", output.shape)
    except Exception as e:
        print("Error:", e)

    # print(summary(model, input_size=(1, 1, 512, 16)))
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Input Data Paths and Output Save Paths

    # Load Dataset and Feed to Dataloader
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_07262024_0to1/test/"

    pulse_specification = None


    data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=1, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=1)

    # data_val = DataMilking_MilkCurds(root_dirs=[datapath_val], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=3)

    print(len(data_test))
    # Calculate the lengths for each split
    train_size = 0*int(0.8 * len(data_test))
    val_size = 0*int(0.2 * len(data_test))
    test_size = len(data_test) - train_size - val_size
    #print sizes of train, val, and test
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data_test, [train_size, val_size, test_size])
    
    # Create data loaders
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require gradients!")

    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08142024_regressionResnet18_5/evaluate_outputs"
    best_model_regression_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08142024_regressionResnet18_5/resNetregression_18_2000classes_best_model.pth"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    identifier = "resNetregression_18_2000classes"
    criterion = nn.MSELoss()
    state_dict = torch.load(best_model_regression_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(state_dict)
    test_model(model, test_dataloader, model_save_dir, identifier, device, criterion=criterion, denoising=False, denoise_model =None,
                zero_mask_model = None, parallel=True, num_classes=num_classes)
    # print(summary(model=model, 
    #     input_size=(32, 1, 16, 512), # make sure this is "input_size", not "input_shape"
    #     # col_names=["input_size"], # uncomment for smaller output
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"]
    # ))
    print(summary(model, input_size=(1, 1, 512, 16)))
if __name__ == "__main__":
    main()