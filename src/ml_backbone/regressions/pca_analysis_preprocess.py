from regression_util import *
from sklearn.decomposition import PCA
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
    datapath_train = "/sdf/data/lcls/ds/prj/prjs2e21/results/1-Pulse_03282024/Processed_07262024_0to1/train/"

    pulse_specification = None


    data_train = DataMilking_MilkCurds(root_dirs=[datapath_train], input_name="Ypdf", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=1, zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=1)
    # data_train_2 = DataMilking_MilkCurds(root_dirs=[datapath_train], input_name="Ximg", pulse_handler=None, transform=None, pulse_threshold=4, test_batch=1,zero_to_one_rescale=False, phases_labeled=True, phases_labeled_max=1)

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

    # # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(data_train, [train_size, val_size, test_size])
    # train_size = int(0.8 * len(data_train_2))
    # val_size = int(0.2 * len(data_train_2))
    # test_size = len(data_train_2) - train_size - val_size
    # #print sizes of train, val, and test
    # print(f"Train size: {train_size}")
    # print(f"Validation size: {val_size}")
    # print(f"Test size: {test_size}")
    # train_dataset_2, val_dataset_2, test_dataset_2 = random_split(data_train_2, [train_size, val_size, test_size])



    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # train_dataloader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True)
    # val_dataloader_2 = DataLoader(val_dataset_2, batch_size=32, shuffle=False)
    # test_dataloader_2 = DataLoader(test_dataset_2, batch_size=32, shuffle=False)

    model_save_dir = "/sdf/data/lcls/ds/prj/prjs2e21/results/COOKIE_ML_Output/regression/run_08082024_regressionSingleLSTMTest_2/"
    # Check if directory exists, otherwise create it
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    n_components = 512*16
    def apply_pca(train_loader, n_components=n_components):
        flattened_images = []
        
        for images, temp1, temp2 in train_loader:
            images = images.view(images.size(0), -1)  # Flatten the images
            flattened_images.append(images)
        
        flattened_images = torch.cat(flattened_images)
        flattened_images = flattened_images.numpy()
        
        pca = PCA(n_components=n_components)
        pca.fit(flattened_images)
        
        transformed_images = pca.transform(flattened_images)
        return transformed_images, pca

# Apply PCA to the training images
    train_pca, pca_model = apply_pca(train_dataloader, n_components=100)
    print(train_pca.shape)

    eigenvalues = pca_model.explained_variance_
    print(eigenvalues)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, n_components + 1), eigenvalues, 'o-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue (Explained Variance)')
    plt.show()
    plt.savefig('scree_plot.png')

    # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # # Plot cumulative explained variance
    # plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(1, n_components + 1), cumulative_variance, 'o-', linewidth=2)
    # plt.title('Cumulative Explained Variance')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
