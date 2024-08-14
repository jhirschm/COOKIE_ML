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
def get_phase(outputs, num_classes, max_val=2*torch.pi):
    # Step 1: Convert the model outputs to probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)

    # Step 2: Get the index of the highest probability class
    max_prob_idx = torch.argmax(probabilities, dim=1)

    # Step 3: Map the class index to a phase value between 0 and 2*pi
    phase_values = max_prob_idx * (max_val / num_classes)

    return phase_values

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, 
                    checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=100, denoising=False,
                    denoise_model=None, zero_mask_model=None, parallel=True,
                    second_denoising=False, second_train_dataloader=None, second_val_dataloader=None, num_classes=1000):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        start_epoch = 0
        if denoising and denoise_model is None and zero_mask_model is None:
            raise ValueError("Denoising is enabled but no denoising model is provided")
        if parallel:
            model = nn.DataParallel(model)
            if denoising and denoise_model is not None and zero_mask_model is not None:
                denoise_model = nn.DataParallel(denoise_model)
                zero_mask_model = nn.DataParallel(zero_mask_model)
                denoise_model.to(device)
                zero_mask_model.to(device)
        model.to(device)
        checkpoint_path = os.path.join(model_save_dir, f"{identifier}_checkpoint.pth")

        
        
        # Try to load from checkpoint if it exists and resume_from_checkpoint is True
        if checkpoints_enabled and resume_from_checkpoint and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint['best_epoch']
            best_model = None
        name = f"{model_save_dir}/{identifier}" + "_run_time_info.txt"
        with open(name, "a") as f:
            f.write(f"Training resumed at {datetime.datetime.now()} from epoch {start_epoch}\n" if start_epoch > 0 else f"Training started at {datetime.datetime.now()}\n")

            for epoch in range(start_epoch, max_epochs):
                model.train()  # Set the model to training mode
                running_train_loss = 0.0


                for batch in train_dataloader:
                    optimizer.zero_grad()  # Zero the parameter gradients

                    inputs, labels, phases = batch
                    inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                    print(inputs.shape)
                    # phases = phases.to(dtype)
                    # print(labels)
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
                    print(outputs)
          
                    outputs = get_phase(outputs, num_classes, max_val=2*torch.pi)
                    print(outputs)
                    print(phases)
                    loss = criterion(outputs, phases)
                    print(loss)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()
                if second_train_dataloader is not None and second_denoising:
                    for batch in train_dataloader:
                        optimizer.zero_grad()  # Zero the parameter gradients

                        inputs, labels, phases = batch
                        inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                        # phases = phases.to(model.module.dtype)
                        # print(labels)
                        if second_denoising and denoise_model is not None and zero_mask_model is not None:
                        
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
                        loss = criterion(outputs, phases)
                        loss.backward()
                        optimizer.step()

                    running_train_loss += loss.item()

                # train_loss = running_train_loss / (len(train_dataloader) + (len(second_train_dataloader) if second_train_dataloader else 0))
                train_loss = running_train_loss / (len(train_dataloader) + (len(second_train_dataloader) if second_train_dataloader else 0))

                train_losses.append(train_loss)
               

                # Validation loop
                model.eval()  # Set the model to evaluation mode
                running_val_loss = 0.0

                with torch.no_grad():
                    for batch in val_dataloader:
                        inputs, labels, phases = batch
                        inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                        # phases = phases.to(model.module.dtype)
                        # print(labels)
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
                        loss = criterion(outputs, phases)
                        running_val_loss += loss.item()
                    
                    if second_val_dataloader is not None and second_denoising:
                        for batch in val_dataloader:

                            inputs, labels, phases = batch
                            inputs, labels, phases = inputs.to(device), labels.to(device), phases.to(device)
                            # phases = phases.to(model.module.dtype)
                            # print(labels)
                            if second_denoising and denoise_model is not None and zero_mask_model is not None:
                            
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

                            # loss = ((torch.cos(outputs*2*np.pi)-torch.cos(phases_differences*2*np.pi))**2 + (torch.sin(outputs*2*np.pi)-torch.sin(phases_differences*2*np.pi))**2).mean()
                            loss = criterion(outputs, phases)
                            running_val_loss += loss.item()
            
                val_loss = running_val_loss / (len(val_dataloader) + (len(second_val_dataloader) if second_val_dataloader else 0))
                # val_loss = running_val_loss / (len(val_dataloader) + (len(second_val_dataloader) if second_val_dataloader else 0))

                val_losses.append(val_loss)

                f.write(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")
                print(f"Epoch [{epoch+1}/{max_epochs}] - Train Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}\n\n")

                # Update the scheduler
                should_stop = scheduler.step(val_loss, epoch)

                # Check if this is the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model = model.state_dict().copy()

                    # Save the best model with a specified name and path in the model_dir
                    best_model_path = f"{model_save_dir}/{identifier}_best_model.pth"
                    torch.save(model.state_dict(), best_model_path)

                ## Save checkpoint
                if checkpoints_enabled:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'best_val_loss': best_val_loss,
                        'best_epoch': best_epoch,
                    }
                    torch.save(checkpoint, checkpoint_path)
                
                # Early stopping check
                # if scheduler.should_stop():
                #     print(f"Early stopping at epoch {epoch+1}")
                #     break
                if should_stop:
                    print(f"Early stopping at epoch {epoch+1}\n")
                    f.write(f"Early stopping at epoch {epoch+1}\n")
                    break
                f.flush() # Flush the buffer to write to the file
        # Save the output to the specified file
        run_summary_path = f"{model_save_dir}/{identifier}"+ "_run_summary.txt"
        with open(run_summary_path, "w") as file:
            file.write("Number of Epochs for Best Model: {}\n".format(best_epoch + 1))
            file.write("Final Training Loss: {:.10f}\n".format(train_losses[-1]))
            file.write("Final Validation Loss: {:.10f}\n".format(val_losses[-1]))

        # Plot the training and validation losses
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.scatter(best_epoch, val_losses[best_epoch], marker='*', color='red', label='Best Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        losses_path = os.path.join(model_save_dir, identifier + "_losses.pdf")
        plt.savefig(losses_path)
        plt.close()

        return best_model, best_epoch, train_losses[-1], val_losses[-1], best_val_loss

def main():
    
    dtype = torch.float32


   
    fake_input = torch.randn(1, 1, 512, 16, device=device, dtype=dtype)
    
    # model = ResNet(block=BasicBlock, layers=[2,2,1,1], num_classes=1000)
    num_classes = 1000
    model = resnet34(num_classes=num_classes)
    model = model.to(device).to(dtype)


    try:
        output = model(fake_input)
        print("Output shape:", output.shape)
    except Exception as e:
        print("Error:", e)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Input Data Paths and Output Save Paths

    # Load Dataset and Feed to Dataloader
    datapath_train = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_07262024_0to1/train/"

    pulse_specification = None


    data_train = DataMilking_MilkCurds(root_dirs=[datapath_train], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=1, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=1)

    # data_val = DataMilking_MilkCurds(root_dirs=[datapath_val], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=3)

    print(len(data_train))
    # Calculate the lengths for each split
    train_size = int(0.8 * len(data_train))
    val_size = int(0.2 * len(data_train))
    test_size = len(data_train) - train_size - val_size
    #print sizes of train, val, and test
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data_train, [train_size, val_size, test_size])
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08132024_regressionResnet34_1"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    max_epochs = 200
    scheduler = CustomScheduler(optimizer, patience=3, early_stop_patience = 10, cooldown=2, lr_reduction_factor=0.5, max_num_epochs = max_epochs, improvement_percentage=0.001)

    identifier = "resNetregression"

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, model_save_dir, identifier, device, 
                                 checkpoints_enabled=True, resume_from_checkpoint=False, max_epochs=max_epochs, denoising=False, 
                                 denoise_model =None , zero_mask_model = None, parallel=True, second_denoising=False, num_classes=num_classes)
if __name__ == "__main__":
    main()