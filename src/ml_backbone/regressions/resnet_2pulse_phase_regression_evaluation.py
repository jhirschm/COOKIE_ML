from regression_util import *
from pulse_phase_regression import RegressionModel
from skimage.transform import iradon
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, multilabel_confusion_matrix, accuracy_score
import seaborn as sns




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

def decode_2hot_to_phases(output_vector, n_classes, phase_range=(0, 2 * np.pi)):
    """
    Decodes the 2-hot encoded output vector into the original phase values.

    Args:
    output_vector (torch.Tensor): The prediction vector (shape: [batch_size, n_classes]).
    n_classes (int): Number of classes (must match the encoder's n_classes).
    phase_range (tuple): Tuple indicating the range of phase values (min_phase, max_phase).

    Returns:
    phases1 (torch.Tensor): Decoded first phase values (shape: [batch_size]).
    phases2 (torch.Tensor): Decoded second phase values (shape: [batch_size]).
    """
    min_phase, max_phase = phase_range

    # Apply sigmoid to the output vector
    # print(output_vector)
    probabilities = 1/(1+np.exp(-1*(output_vector)))
    # print(probabilities)

    
    # Split the vector into two halves
    half_n_classes = n_classes // 2
    first_half = probabilities[:, :half_n_classes]
    second_half = probabilities[:, half_n_classes:]

    # Get the max index from each half (corresponding to the predicted phases)
    idx1 = np.argmax(first_half, axis=1)
    idx2 = np.argmax(second_half, axis=1)

    # Convert indices back to phase values
    phases1 = (idx1.astype(float) / half_n_classes) * (max_phase - min_phase) + min_phase
    phases2 = (idx2.astype(float) / half_n_classes) * (max_phase - min_phase) + min_phase

    output_phases = np.ndarray((phases1.shape[0], 2))
    output_phases[:, 0] = phases1
    output_phases[:, 1] = phases2
    return output_phases

def earth_mover_distance(y_pred, y_true):
    """
    Calculates the Earth Mover's Distance (EMD) between the cumulative sums
    of the predicted and true distributions.

    Args:
    y_pred (torch.Tensor): The predicted logits or probabilities.
    y_true (torch.Tensor): The true distributions (typically one-hot encoded).

    Returns:
    torch.Tensor: The computed EMD loss.
    """
    # Ensure both inputs are probabilities
    y_pred_prob = torch.sigmoid(y_pred)
    
    # Compute the cumulative sums
    cumsum_y_true = torch.cumsum(y_true, dim=-1)
    cumsum_y_pred = torch.cumsum(y_pred_prob, dim=-1)
    
    # Calculate the squared difference between cumulative sums
    emd_loss = torch.mean(torch.square(cumsum_y_true - cumsum_y_pred), dim=-1)
    
    return torch.mean(emd_loss)

def phase_to_2hot(phases1, phases2, n_classes, phase_range=(0, 2 * torch.pi)):
    """
    Converts batches of phase values into a batch of 2-hot encoded vectors.

    Args:
    phases1 (torch.Tensor): Batch of first phase values (shape: [batch_size]).
    phases2 (torch.Tensor): Batch of second phase values (shape: [batch_size]).
    n_classes (int): Number of classes for the 2-hot encoding.
    phase_range (tuple): Tuple indicating the range of phase values (min_phase, max_phase).

    Returns:
    torch.Tensor: Batch of 2-hot encoded vectors (shape: [batch_size, n_classes]).
    """
    min_phase, max_phase = phase_range

    # Ensure the phases are within the specified range
    assert torch.all((min_phase <= phases1) & (phases1 <= max_phase)), f"Some values in phases1 are out of the specified range ({min_phase}, {max_phase}). Values: {phases1}"
    assert torch.all((min_phase <= phases2) & (phases2 <= max_phase)), f"Some values in phases2 are out of the specified range ({min_phase}, {max_phase}). Values: {phases2}"

    # Normalize the phases to a range from 0 to n_classes
    phases1_norm = (phases1 - min_phase) / (max_phase - min_phase)
    phases2_norm = (phases2 - min_phase) / (max_phase - min_phase)

    # Convert normalized phases to class indices for each half of the vector
    half_n_classes = n_classes // 2
    idx1 = (phases1_norm * half_n_classes).long() % half_n_classes
    idx2 = (phases2_norm * half_n_classes).long() % half_n_classes + half_n_classes

    # Create 2-hot encoded vectors
    batch_size = phases1.size(0)
    one_hot_vectors = torch.zeros(batch_size, n_classes, device=phases1.device)
    one_hot_vectors[torch.arange(batch_size), idx1] = 1
    one_hot_vectors[torch.arange(batch_size), idx2] = 1

    return one_hot_vectors

def phases_to_1hot(phase, num_classes, phase_range=(0, 2 * torch.pi)):
    '''
    Converts a batch of phase values to a batch of one-hot encoded vectors.
    '''
    min_phase, max_phase = phase_range
    phases_norm = (phase - min_phase) / (max_phase - min_phase)
    idx = (phases_norm * num_classes).long() % num_classes
    one_hot_vectors = torch.zeros(phase.size(0), num_classes, device=phase.device)
    one_hot_vectors[torch.arange(phase.size(0)), idx] = 1
    return one_hot_vectors

def phases_to_1hot_wrapping(phase, num_classes, phase_range=(0, torch.pi)):
    '''
    Converts a batch of phase values to a batch of one-hot encoded vectors.
    '''
    min_phase, max_phase = phase_range
    # phases = torch.fmod(torch.abs(phase), torch.pi)
    phases = torch.arccos(torch.cos(phase)) #not including sign now
    phases_norm = (phases - min_phase) / (max_phase - min_phase)

    idx = (phases_norm * num_classes).long() % num_classes
    one_hot_vectors = torch.zeros(phase.size(0), num_classes, device=phase.device)
    one_hot_vectors[torch.arange(phase.size(0)), idx] = 1
    return one_hot_vectors

def onehot_to_phase(one_hot_vector, num_classes, phase_range=(0, 2 * torch.pi)):
    '''
    Converts a batch of one-hot encoded vectors to a batch of phase values.
    '''
    min_phase, max_phase = phase_range
    idx = torch.argmax(one_hot_vector, dim=1)
    phases_norm = idx.float() / num_classes
    phases = phases_norm * (max_phase - min_phase) + min_phase
    return phases

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


def test_model(model, test_dataloader, model_save_dir, identifier, device, denoising=False,criterion=None,
               denoise_model=None, zero_mask_model=None, parallel=True, num_classes=1000, inverse_radon=False, multi_hotEncoding_eval=False, top_n_classes=10, phase_dif_pred=False, phase_1hotwrapping=False, phase_mispredict_analysis=False,
               lstm_classifier_cases=False, lstm_model=None):
    test_losses = []
    true_phase_list = []
    predicted_phase_list = []
    inputs_list = []
    ypdfs_list = []
    denoised_inputs_list = []
    running_test_loss = 0
    true_pulses = []
    predicted_pulses = []
    model.to(device)

    if denoising and denoise_model is None and zero_mask_model is None:
        raise ValueError("Denoising is enabled but no denoising model is provided")
    if parallel:
        model = nn.DataParallel(model)
        if denoising and denoise_model is not None and zero_mask_model is not None:
            denoise_model = nn.DataParallel(denoise_model)
            zero_mask_model = nn.DataParallel(zero_mask_model)
            denoise_model.to(device)
            zero_mask_model.to(device)
    checkpoint_path = os.path.join(model_save_dir, f"{identifier}_checkpoint.pth")

    if inverse_radon:
        n = 16
        theta = np.linspace(0., 360., n, endpoint=False)

        
        
    model.eval()  # Set the model to evaluation mode
  
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, phases, ypdfs = batch
            inputs, labels, phases, ypdfs = inputs.to(device), labels.to(device), phases.to(device), ypdfs.to(device)
            if denoising and denoise_model is not None and zero_mask_model is not None:
                        
                denoise_model.eval()
                zero_mask_model.eval()
                
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device, torch.float32)
                # labels = labels[0]
                
                outputs = denoise_model(inputs)
                if phase_mispredict_analysis:
                    inputs_cpu = inputs.cpu().detach()
                    outputs_cpu = outputs.cpu().detach()
                    ypdfs_cpu = ypdfs.cpu().detach()

                    for i in range(inputs_cpu.size(0)):  # Loop over batch size (32 in this case)
                        inputs_list.append(inputs_cpu[i].squeeze())  # Remove extra channel if needed, making it [512, 16]
                        denoised_inputs_list.append(outputs_cpu[i].squeeze())  # Same for outputs         
                        ypdfs_list.append(ypdfs_cpu[i].squeeze())  # Same for outputs
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
                lstm_inputs = outputs.detach().clone()
                lstm_inputs = lstm_inputs.to(device)
                inputs = torch.unsqueeze(outputs, 1)
                inputs = inputs.to(device, torch.float32)


            else: 
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device, torch.float32)
                            
            if inverse_radon:
                recon_images = []
                transpose_inputs = torch.transpose(inputs, 2, 3)
                for i in range(transpose_inputs.size(0)):  # Iterate over the batch
                    # Extract the image and remove the channel dimension for processing with skimage
                    image_np = transpose_inputs[i, 0].cpu().numpy()  # shape: [height, width]

                    # Compute the inverse Radon transform
                    recon_image_np = iradon(image_np, theta=theta, filter_name='ramp', circle=False)
                    
                    # Convert back to a PyTorch tensor and add channel dimension back
                    recon_image_tensor = torch.tensor(recon_image_np, dtype=torch.float32).unsqueeze(0)  # shape: [1, height, width]

                    # Add to the list of reconstructed images
                    recon_images.append(recon_image_tensor)
                # Stack the reconstructed images back into a batch
                recon_images = torch.stack(recon_images)

                # Add the batch dimension back
                recon_images = recon_images.to(device)  # shape: [batch_size, 1, height, width]
                inputs = recon_images
            outputs = model(inputs).to(device)
            if multi_hotEncoding_eval:
                phases = phases.to(torch.float32)
                phases_encoded = phase_to_2hot(phases[:,0], phases[:,1], num_classes)
                phases_encoded = torch.tensor(phases_encoded).to(device)

                # Store the true and predicted values
                true_phase_list.append(phases_encoded.cpu().numpy())
                predicted_phase_list.append(outputs.cpu().numpy())

                # Calculate test loss (using binary cross entropy)
                loss = criterion(outputs, phases_encoded)
                running_test_loss += loss.item()
            elif phase_dif_pred:
                phases = phases.to(torch.float32)
                phases_dif = phases[:,0] - phases[:,1]
                output_dif = get_phase(outputs, num_classes, max_val=2*torch.pi)
                loss = criterion(output_dif, phases_dif)
                true_phase_list.append(phases_dif.cpu().numpy())
                predicted_phase_list.append(output_dif.cpu().numpy())
                predicted_phase_list= np.array(predicted_phase_list)
                true_phase_list = np.array(true_phase_list)
            elif phase_1hotwrapping:
                phases = phases.to(torch.float32)
                phases_dif = phases[:,0] - phases[:,1]
                phases_one_hot = phases_to_1hot_wrapping(phases_dif, num_classes, phase_range=(0, torch.pi))
                predicted_phases_decoded = onehot_to_phase(outputs, num_classes, phase_range=(0, torch.pi))
                # print("Predicted Phases Decoded:", predicted_phases_decoded)
                # loss = criterion(outputs, phases_one_hot)

                true_phase_list.append(phases_dif.cpu().numpy().ravel())
                predicted_phase_list.append(predicted_phases_decoded.cpu().numpy().ravel())


            if lstm_classifier_cases:
                probs, preds = lstm_model.predict(lstm_inputs)
                preds = preds.to(device)
                probs = probs.to(device)
    
                predicted_pulse_single_label = np.argmax(probs.cpu().numpy(), axis=1)
                predicted_pulses.extend(predicted_pulse_single_label)

            # else:
            #     outputs_1 = get_phase(outputs[:,0:outputs.shape[1]//2], num_classes//2, max_val=2*torch.pi)
            #     outputs_2 = get_phase(outputs[:,outputs.shape[1]//2:], num_classes//2, max_val=2*torch.pi)
                
            #     phases = phases.to(torch.float32)
            #     phases_1 = phases[:,0:1]
            #     phases_2 = phases[:,1:2]
            #     if i == 1:
            #         print("***********  Validation  ***********")
            #         print(outputs_1)
            #         print(outputs_2)  
            #         print(phases_1)
            #         print(phases_2)  
            #         print("***********  ***  ***********")
            #         i+=1
            #     loss_1 = criterion(outputs_1, phases_1)
            #     loss_2 = criterion(outputs_2, phases_2)
            #     loss_a = loss_1 + loss_2
            #     loss_1 = criterion(outputs_1, phases_2)
            #     loss_2 = criterion(outputs_2, phases_1)
            #     loss_b = loss_1 + loss_2
            #     output_dif = get_phase(outputs, num_classes, max_val=2*torch.pi)
            #     phases_differences = phases_1 - phases_2
            #     loss_c =  ((torch.cos(output_dif)-torch.cos(phases_differences))**2 + (torch.sin(output_dif)-torch.sin(phases_differences))**2).mean()
            #     loss = torch.min(loss_a, loss_b)
            #     loss = loss_c
                        
            # After looping over the batches, flatten the lists
        # true_phases = np.vstack(true_phase_list)
        # predicted_phases = np.vstack(predicted_phase_list)

        # print("Shapes")
        # print(predicted_phases.shape)
        # print(true_phases.shape)

        # # Convert predicted logits to binary predictions
        # predicted_phases_binary = (predicted_phases > 0.5).astype(int)
        
        # # Calculate evaluation metrics
        # exact_match_ratio = accuracy_score(true_phases, predicted_phases_binary)
        # hamming_loss_value = hamming_loss(true_phases, predicted_phases_binary)
        # precision = precision_score(true_phases, predicted_phases_binary, average='macro')
        # recall = recall_score(true_phases, predicted_phases_binary, average='macro')
        # f1 = f1_score(true_phases, predicted_phases_binary, average='macro')
        
        # print(f"Exact Match Ratio: {exact_match_ratio:.4f}")
        # print(f"Hamming Loss: {hamming_loss_value:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 Score: {f1:.4f}")
        
        # # Calculate multilabel confusion matrix
        # multilabel_cm = multilabel_confusion_matrix(true_phases, predicted_phases_binary)
        
        # # Aggregate confusion matrices into a summary (optional, depends on your use case)
        # summary_cm = np.sum(multilabel_cm, axis=0)
        
        # # Plot confusion matrix for top N classes
        # top_n_classes = min(top_n_classes, num_classes)
        # for i in range(top_n_classes):
        #     cm = multilabel_cm[i]
        #     plt.figure(figsize=(10, 7))
        #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        #     plt.title(f'Confusion Matrix for Class {i}')
        #     plt.xlabel('Predicted')
        #     plt.ylabel('True')
        #     plt.savefig(os.path.join(model_save_dir, f"{identifier}_confusion_matrix_class_{i}.png"))
        #     plt.close()

    # predicted_phases_decoded = decode_2hot_to_phases(predicted_phases, predicted_phases.shape[1], phase_range=(0,2*np.pi))
    # true_phases_decoded = decode_2hot_to_phases(true_phases, predicted_phases.shape[1], phase_range=(0,2*np.pi))

    # predicted_phase_difference = predicted_phases_decoded[:, 0] - predicted_phases_decoded[:, 1]
    # true_phase_difference = true_phases_decoded[:, 0] - true_phases_decoded[:, 1]

    # Calculate and print the average test loss
    true_phase_list = np.concatenate(true_phase_list, axis=0)  # Shape should now be (8192,)
    predicted_phase_list = np.concatenate(predicted_phase_list, axis=0)  # Shape should now be (8192,)
    print("True Phase List:", true_phase_list)
    print("Predicted Phase List:", predicted_phase_list)
    plot_path = os.path.join(model_save_dir, identifier + "_TruePred.pdf")
    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.scatter(np.abs(true_phase_list), np.abs(predicted_phase_list), color='blue', label='Predicted vs True')
    plt.plot([np.abs(true_phase_list).min(), np.abs(true_phase_list).max()], 
            [np.abs(true_phase_list).min(), np.abs(true_phase_list).max()], 
            color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel('True Abs Phase Differences')
    plt.ylabel('Predicted Abs Phase Differences')
    plt.title('True vs Predicted Phase Differences')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(plot_path)

    plot_path = os.path.join(model_save_dir, identifier + "_ArccosCosTruePred.pdf")
    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arccos(np.cos(true_phase_list)), predicted_phase_list, color='blue', label='Predicted vs True')
    plt.plot([np.arccos(np.cos(true_phase_list)).min(), np.arccos(np.cos(true_phase_list)).max()], 
            [np.arccos(np.cos(true_phase_list)).min(), np.arccos(np.cos(true_phase_list)).max()], 
            color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel('True ArccosCos Phase Differences')
    plt.ylabel('Predicted Phase Differences')
    plt.title('ArccosCos True vs Predicted Phase Differences')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(plot_path)

    if lstm_classifier_cases:
        plot_path = os.path.join(model_save_dir, identifier + "_ArccosCosTruePred_LSTMClassifier.pdf")
        cmap = cm.get_cmap('viridis', 5)  # 5 distinct colors for categories 0-4

            # Create the scatter plot with colors based on predicted_pulses
        plt.figure(figsize=(10, 6))

        # Scatter plot with color-coded points
        scatter = plt.scatter(np.arccos(np.cos(true_phase_list)), 
                            predicted_phase_list, 
                            c=predicted_pulses, 
                            cmap=cmap, 
                            label='Predicted vs True', 
                            s=50, edgecolor='k', alpha=0.75)
        # plt.scatter(np.arccos(np.cos(true_phase_list)), predicted_phase_list, color='blue', label='Predicted vs True')
        # print("Predicted Pulses:", predicted_pulses[0:100])
        # print("Pred Phases:", predicted_phase_list[0:100])
        # print("True Phases:", np.arccos(np.cos(true_phase_list[0:100])))


        # # Add colorbar to show the mapping of colors to categories
        cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(['0', '1', '2', '3', '4+'])
        cbar.set_label('LSTM Classifier Categories')

        # Plot the ideal prediction line
        plt.plot([np.arccos(np.cos(true_phase_list)).min(), np.arccos(np.cos(true_phase_list)).max()], 
                [np.arccos(np.cos(true_phase_list)).min(), np.arccos(np.cos(true_phase_list)).max()], 
                color='red', linestyle='--', label='Ideal Prediction')

        plt.xlabel('True ArccosCos Phase Differences')
        plt.ylabel('Predicted Phase Differences')
        plt.title('ArccosCos True vs Predicted Phase Differences')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the plot
        plt.savefig(plot_path) 

        # Calculate the phase difference (true - predicted) for all data points
        phase_diff_all = np.abs(np.arccos(np.cos(true_phase_list)) - predicted_phase_list)

        # Compute mean and standard deviation for the entire dataset
        mean_all = np.mean(phase_diff_all)
        std_all = np.std(phase_diff_all)

        print("--------------------")
        print(f"Mean of phase difference (all data): {mean_all}")
        print(f"Standard deviation of phase difference (all data): {std_all}")

        # Now filter for data points classified as "2 pulses" (3rd index, value = 2 in predicted_pulses)
        predicted_pulses = int(predicted_pulses)
        print(predicted_pulses)
        mask_two_pulses = (predicted_pulses == 2)

        # Apply the mask to get the phase differences only for the points classified as "2 pulses"
        phase_diff_two_pulses = phase_diff_all[mask_two_pulses]

        # Compute mean and standard deviation for the "2 pulses" subset
        mean_two_pulses = np.mean(phase_diff_two_pulses)
        std_two_pulses = np.std(phase_diff_two_pulses)
        print("--------------------")

        print(f"Mean of phase difference (2 pulses): {mean_two_pulses}")
        print(f"Standard deviation of phase difference (2 pulses): {std_two_pulses}")             

    plot_path = os.path.join(model_save_dir, identifier + "_SinTruePred.pdf")
    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.scatter(np.sin((true_phase_list)), np.sin((predicted_phase_list)), color='blue', label='Predicted vs True')
    # plt.plot([true_phase_differences_array.min(), true_phase_differences_array.max()], 
    #         [true_phase_differences_array.min(), true_phase_differences_array.max()], 
    #         color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel('Sin True Abs Phase Differences')
    plt.ylabel('Sin Predicted Abs Phase Differences')
    plt.title('Sin Abs True vs Predicted Phase Differences')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(plot_path)

    plot_path = os.path.join(model_save_dir, identifier + "_CosTruePred.pdf")

    plt.figure(figsize=(10, 6))
    plt.scatter(np.cos((true_phase_list)), np.cos((predicted_phase_list)), color='blue', label='Predicted vs True')
    # plt.plot([true_phase_differences_array.min(), true_phase_differences_array.max()], 
    #         [true_phase_differences_array.min(), true_phase_differences_array.max()], 
    #         color='red', linestyle='--', label='Ideal Prediction')
    plt.xlabel('Cos True Abs Phase Differences')
    plt.ylabel('Cos Predicted Abs Phase Differences')
    plt.title('Cos Abs True vs Predicted Phase Differences')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(plot_path)

    if phase_mispredict_analysis:
        input_list = np.array(inputs_list)
        denoised_input_list = np.array(denoised_inputs_list)

        #calculate residuals between arccos(cos(phase)) and arccos(cos(predicted))
        residuals = np.arccos(np.cos(true_phase_list)) - predicted_phase_list
        residuals = residuals.flatten()

        # plot residuals vs input
        plot_path = os.path.join(model_save_dir, identifier + "_ResidualsVsInput.pdf")    
        plt.figure(figsize=(10, 6))
        plt.scatter(np.arccos(np.cos(true_phase_list)).flatten(), residuals, color='blue', label='Residuals vs Input')
        plt.xlabel('Input') 
        plt.ylabel('Residuals')
        plt.title('Residuals vs Input')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)

        # For the predicted phases that are close extremely close to zero, plot the sinogram  from input list and denoised input list a add as text the arc cos cos of the true phase the output predicted phaae and the residual
        # Make
        save_dir = os.path.join(model_save_dir, identifier + "_SinogramResiduals")
        # Define your threshold for detecting near-zero predicted phases
        epsilon = 0.05

        # Directory to save plots
        os.makedirs(save_dir, exist_ok=True)

        # Assuming you have these lists filled
        
        # Iterate over the sinograms and their corresponding predicted and true phases
        # print("Input List:", input_list.shape)
        # print("Denoised Input List:", denoised_input_list.shape)
        # print("Predicted Phase List:", predicted_phase_list.shape)
        # print("True Phase List:", true_phase_list.shape)
        for idx, (input_sino, denoised_sino, ypdf_sino, predicted_phase, true_phase) in enumerate(zip(inputs_list, denoised_inputs_list, ypdfs_list, predicted_phase_list, true_phase_list)):
            # print("Predicted Phase:", predicted_phase.shape)
            # print("True Phase:", true_phase.shape)
            true_phase_adjusted = np.arccos(np.cos(true_phase))

            # Check for cases where predicted phase is nearly zero but true phase isn't
            if abs(predicted_phase) < epsilon and true_phase_adjusted > epsilon:
                
                # Calculate arccos(cos(true_phase)) for the label

                # Plot the two sinograms (initial input and denoised input)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(ypdf_sino.cpu().numpy(), cmap=plt.get_cmap('Blues'), aspect='auto')
                axes[0].set_title("Ypdf Sinogram")
                axes[0].axis('off')

                # Plot the initial input sinogram
                axes[1].imshow(input_sino.cpu().numpy(), cmap=plt.get_cmap('Blues'), aspect='auto')
                axes[1].set_title("Initial Input Sinogram")
                axes[1].axis('off')
                
                # Plot the denoised input sinogram
                axes[2].imshow(denoised_sino.cpu().numpy(), cmap=plt.get_cmap('Blues'), aspect='auto')
                axes[2].set_title("Denoised Input Sinogram")
                axes[2].axis('off')
                
                # Add text annotations with the predicted and true phase
                fig.suptitle(f"Predicted Phase: {predicted_phase:.4f}, Adjusted True Phase: {true_phase_adjusted:.4f}", fontsize=12)

                # Save the plot in the specified directory
                plot_filename = os.path.join(save_dir, f"sinogram_plot_{idx}.png")
                plt.savefig(plot_filename, bbox_inches='tight')
                
                # Close the plot to avoid memory issues
                plt.close(fig)



                
                


    return 1
def main():
    
    dtype = torch.float32


   
    fake_input = torch.randn(1, 1, 512, 16, device=device, dtype=dtype)
    # fake_input = torch.randn(1, 1, 362, 362, device=device, dtype=dtype)


    
    # model = ResNet(block=BasicBlock, layers=[2,2,1,1], num_classes=1000)
    num_classes = 4000#1024
    # model = resnet152(num_classes=num_classes)
    # model = resnet34(num_classes=num_classes)
    # model = resnet50(num_classes=num_classes)
    model = resnet34(num_classes=num_classes)

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
    datapath_test = "/sdf/data/lcls/ds/prj/prjs2e21/results/2-Pulse_04232024/Processed_07312024_0to1/test/"

    pulse_specification = None


    data_test = DataMilking_MilkCurds(root_dirs=[datapath_test], input_name="Ximg", pulse_handler=None, test_batch=1, transform=None, pulse_threshold=4, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=2, ypdfs_included=True)

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

    # best_model_regression_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08282024_Resnext34_2hotsplit_EMDloss_Ypdf_1/Resnext34_2hotsplit_EMDloss_Ypdf_best_model.pth"
    best_model_regression_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08302024_Resnext34_dif_Ypdf_1/Resnext34_dif_Ypdf_3_wrapping_best_model.pth"
    best_model_regression_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08302024_Resnext34_dif_Ximg_1/Resnext34_dif_Ximg_3_wrapping_3_best_model.pth"
    best_model_regression_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_09022024_Resnext34_dif_Ximg_Denoised_1/Resnext34_dif_XimgDenoised_wrapping_best_model.pth"
    best_model_regression_path  = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_09082024_Resnext34_dif_Ximg_Denoised_2/Resnext34_dif_XimgDenoised_wrapping_4_best_model.pth"
    # model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08282024_Resnext34_2hotsplit_EMDloss_Ypdf_1/evaluate_outputs/"
    # model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08302024_Resnext34_dif_Ximg_1/evaluate_outputs/"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_09022024_Resnext34_dif_Ximg_Denoised_1/evaluate_outputs/"
    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_09082024_Resnext34_dif_Ximg_Denoised_2/evaluate_outputs3/"
    # model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_09022024_Resnext34_dif_Ximg_Denoised_1/evaluate_outputs/"

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # identifier = "Resnext34_2hotsplit_EMDloss_Ypdf_train"
    identifier = "Resnext34_dif_Ximg_wrapping_fullData_exrtaAnalysis"
    criterion = nn.BCEWithLogitsLoss()
    state_dict = torch.load(best_model_regression_path, map_location=device)
    state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(state_dict)

    


    '''
    denoising
    '''
    best_autoencoder_model_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07282024_multiPulse/autoencoder_best_model.pth"
    best_model_zero_mask_path = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/denoising/run_07272024_zeroPredict/classifier_best_model.pth"
    
    # Example usage
    encoder_layers = np.array([
        [nn.Conv2d(1, 16, kernel_size=3, padding=2), nn.ReLU()],
        [nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU()]])
   
    decoder_layers = np.array([
        [nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1), nn.ReLU()],
        [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), nn.Sigmoid()]  # Example with Sigmoid activation
        # [nn.ConvTranspose2d(16, 1, kernel_size=3, padding=2), None],  # Example without activation
    ])

    # Example usage
    conv_layers = [
        [nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None],
        [nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()],
        [nn.MaxPool2d(kernel_size=2, stride=2, padding=0), None]
    ]

    output_size = get_conv_output_size((1, 1, 512, 16), conv_layers)

    # Use the calculated size for the fully connected layer input
    fc_layers = [
        [nn.Linear(output_size[1] * output_size[2] * output_size[3], 4), nn.ReLU()],
        [nn.Linear(4, 1), None]
    ]
    zero_model = Zero_PulseClassifier(conv_layers, fc_layers)
    zero_model.to(device)
    state_dict = torch.load(best_model_zero_mask_path, map_location=device)
    keys_to_remove = ['side_network.0.weight', 'side_network.0.bias']
    state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
    zero_model.load_state_dict(state_dict)

    autoencoder = Ximg_to_Ypdf_Autoencoder(encoder_layers, decoder_layers, outputEncoder=False)
    autoencoder.to(device)
    state_dict = torch.load(best_autoencoder_model_path, map_location=device)
    autoencoder.load_state_dict(state_dict)



    data = {
        "hidden_size": 128,#128
        "num_lstm_layers": 3,
        "bidirectional": True,
        "fc_layers": [32, 64],
        "dropout": 0.2,
        "lstm_dropout": 0.2,
        "layerNorm": False,
        # Other parameters are default or not provided in the example
    }   

    # Assuming input_size and num_classes are defined elsewhere
    input_size = 512  # Define your input size
    num_lstm_classes = 5   # Example number of classes

    # Instantiate the CustomLSTMClassifier
    classModel = CustomLSTMClassifier(
        input_size=input_size,
        hidden_size=data['hidden_size'],
        num_lstm_layers=data['num_lstm_layers'],
        num_classes=num_lstm_classes,
        bidirectional=data['bidirectional'],
        fc_layers=data['fc_layers'],
        dropout_p=data['dropout'],
        lstm_dropout=data['lstm_dropout'],
        layer_norm=data['layerNorm'],
        ignore_output_layer=False  # Set as needed based on your application
    )
    classModel.to(device)

    best_mode_classifier = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/lstm_classifier/run_09052024_5classCase/testLSTM_XimgDenoised_best_model.pth"
    state_dict = torch.load(best_mode_classifier, map_location=device)

    state_dict = remove_module_prefix(state_dict)
    # for key in state_dict.keys():
    #     print(key, state_dict[key].shape)
    classModel.load_state_dict(state_dict)

    test_model(model, test_dataloader, model_save_dir, identifier, device, criterion=criterion, denoising=True, denoise_model =autoencoder,
                zero_mask_model = zero_model, parallel=True, num_classes=num_classes, inverse_radon=False, multi_hotEncoding_eval=False, phase_1hotwrapping=True, phase_mispredict_analysis=True,
                lstm_classifier_cases=True, lstm_model=classModel)
    
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