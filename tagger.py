# Import necessary libraries
import torch
import torch.nn as nn
import clip  # OpenAI's CLIP model for image and text embedding
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader


# Custom dataset class to handle image data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, preprocess=None):
        """
        Initialize the dataset with images, optional labels, and preprocessing function.
        :param images: List of images (numpy arrays)
        :param labels: Optional list of labels
        :param preprocess: Optional preprocessing function (e.g., CLIP preprocessing)
        """
        self.images = images
        self.label = labels
        self.preprocess = preprocess

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetch the image and label at the specified index.
        :param idx: Index of the image to retrieve
        :return: Tuple of (image, label)
        """
        image = self.images[idx]
        label = self.label[idx] if self.label is not None else -1
        image = Image.fromarray(image)  # Convert the image to PIL format
        if self.preprocess is not None:
            image = self.preprocess(image)  # Apply preprocessing if provided

        return image, label

# Class to handle tagging with CLIP model based on provided tags
class Tagger():
    def __init__(self, tag_dict):
        """
        Initialize the Tagger with tag dictionary and CLIP model.
        :param tag_dict: Dictionary of tags organized by category
        """
        self.tag_dict = tag_dict
        # Select device based on GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load CLIP model and its preprocessing function
        self.clip_model, self.im_preprocess = clip.load("ViT-B/32", device=self.device)
        # Calculate and store text features for each tag category
        self.text_features = {
            cat: nn.Parameter(tags) for cat, tags in self.get_text_features(tag_dict).items()}

    def get_text_features(self, tag_dict):
        """
        Convert tags to text embeddings using the CLIP model.
        :param tag_dict: Dictionary of tags categorized by type
        :return: Dictionary with category keys and tensor embeddings
        """
        text_features = {}
        with torch.no_grad():
            for category, tags in tag_dict.items():
                # Create descriptive text for each tag
                tag_texts = [f"This image is {tag}" for tag in tags]
                # Tokenize and encode text with CLIP to generate text embeddings
                text_tokens = clip.tokenize(tag_texts).to(self.device)
                text_features[category] = self.clip_model.encode_text(text_tokens).clone().detach().float()

        return text_features

    def create_dataloader(self, images, labels=None):
        """
        Create a DataLoader for batching images.
        :param images: List of images
        :param labels: Optional list of labels
        :return: DataLoader for batched image processing
        """
        dataset = MyDataset(images, labels=labels, preprocess=self.im_preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        return dataloader

    def __call__(self, np_images, threshold=0.23, max_only=True):
        """
        Perform tagging on input images based on similarity to text embeddings.
        :param np_images: List of images in numpy array format
        :param threshold: Minimum similarity score to consider a match
        :param max_only: Whether to consider only the highest score per category
        :return: List of dictionaries with tagging results per image
        """
        # Create DataLoader for input images
        dataloader = self.create_dataloader(np_images)
        resutls = []

        # Process each batch in the DataLoader
        for images, _ in dataloader:
            batch_results = [{} for _ in images]
            with torch.no_grad():
                # Encode images into CLIP image embeddings
                image_features = self.clip_model.encode_image(images.to(self.device)).float()

                # Compare image embeddings to each category's text embeddings
                for category, embedding in self.text_features.items():
                    embedding = embedding.to(self.device)
                    similarity = (image_features @ embedding.T).softmax(dim=-1)

                    # If max_only is true, only save the highest scoring tag above the threshold
                    if max_only:
                        max_vals, max_indices = torch.max(similarity, dim=1)
                        for im_idx, (max_val, max_ind) in enumerate(zip(max_vals, max_indices)):
                            if max_val > threshold:
                                batch_results[im_idx][category] = {
                                    "tag": self.tag_dict[category][max_ind],
                                    "score": max_val.item(),
                                    "index": max_ind.item()}
                    else:
                        # Store all tags above threshold in the batch results
                        for im_idx, similarities in enumerate(similarity):
                            for tag_idx, similarity in enumerate(similarities):
                                if similarity > threshold:
                                    batch_results[im_idx][category] = {
                                        "tag": self.tag_dict[category][tag_idx],
                                        "score": similarity.item(),
                                        "index": tag_idx}

            resutls.extend(batch_results)  # Append batch results to final results
        return resutls

def optimize_model(tagger, images, labels, lr=0.01, epochs=5, print_every=1):
    """
    Fine-tune the tagger model by optimizing text feature embeddings using labeled image data.
    
    :param tagger: Tagger object containing CLIP model and text features
    :param images: List of images to train on
    :param labels: Corresponding labels for each image
    :param lr: Learning rate for the optimizer
    :param epochs: Number of training epochs
    :param print_every: Interval (in epochs) to print training loss
    """
    
    # Set device and retrieve model parameters from the tagger
    device = tagger.device
    model = tagger.text_features  # Text feature embeddings for different tag categories
    
    # Initialize the optimizer with AdamW and set the learning rate
    optimizer = torch.optim.AdamW(model.values(), lr=lr)
    
    # Set up cross-entropy loss, ignoring any labels set to -1 (e.g., for unlabeled data)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    # Create a DataLoader to manage batching of images and labels
    dataloader = tagger.create_dataloader(images, labels)
    
    # Loop through the specified number of training epochs
    for epoch in range(epochs):
        epoch_loss = 0.0  # Track cumulative loss for the epoch
        
        # Process each batch of images and labels from the dataloader
        for t_images, t_labels in dataloader:
            optimizer.zero_grad()  # Clear gradients for the optimizer
            total_loss = torch.tensor(0.0).to(device)  # Initialize total loss for the batch
            
            # Encode images into feature embeddings using the CLIP model
            image_features = tagger.clip_model.encode_image(t_images.to(device)).float()
            
            # Calculate similarity and loss for each category's text embeddings
            for category, embedding in model.items():
                embedding = embedding.to(device)  # Move embeddings to device
                # Calculate similarity between image features and text embeddings
                similarity = image_features @ embedding.T
                # Compute loss for each category's similarity scores and labels
                loss = criterion(similarity, t_labels[category].to(device))
                total_loss += loss  # Accumulate loss across categories

            # Backpropagate the total loss and update model parameters
            total_loss.backward()
            optimizer.step()
            
            # Accumulate the batch loss into the epoch's total loss
            epoch_loss += total_loss.item()
        
        # Print epoch loss at the specified interval
        if epoch % print_every == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss}")



# Generate tags from images using tagger
def create_labels(tagger, images, threshold=0.6):
    """
    Generate labels for images by using the tagger to identify relevant tags above a score threshold.
    This threshold helps filter out low-confidence tags to avoid adding noise.

    :param tagger: Tagger object for generating tags and scores for images
    :param images: List of images to generate labels for
    :param threshold: Minimum score threshold to consider a label valid
    :return: List of dictionaries with labels above the threshold, with noise reduced by marking low scores as -1
    """

    # Generate preliminary labels with scores (with a temporary threshold set to 0)
    tmp_labels = tagger(images, threshold=0)  # Get all scores for analysis

    labels = []  # List to store final labels for all images
    for tmp_label in tmp_labels:
        label = {}  # Dictionary to store labels for the current image

        # Process each category of tags for the image
        for category, tags in tmp_label.items():
            if tags["score"] > threshold:
                # Only include tags with scores above the threshold
                label[category] = tags["index"]
            else:
                # For tags below the threshold, assign label -1 to reduce noise
                label[category] = -1

        labels.append(label)  # Append the processed labels to the final output list

    return labels  # Return the list of labels with noise-reduced tagging
