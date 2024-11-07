# Import necessary libraries
import torch
import torch.nn as nn
import clip  # OpenAI's CLIP model for image and text embedding
from PIL import Image


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

# Optimization Function to Tagger class
def optimize_model(tagger, images, labels, lr=0.01, epochs=5, print_every=1):
    """
    Optimize the tagger model by fine-tuning text embeddings for image tagging.
    :param tagger: Tagger object containing CLIP model and text features
    :param images: List of images to train on
    :param labels: Corresponding labels for each image
    :param lr: Learning rate for the optimizer
    :param epochs: Number of training epochs
    :param print_ever: Frequency (in epochs) to print the loss
    """
    # Set device and model parameters from the tagger
    device = tagger.device
    model = tagger.text_features  # Retrieve text features as model parameters

    # Initialize optimizer and loss criterion
    optimizer = torch.optim.AdamW(model.values(), lr=lr)  # Use AdamW optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Cross-entropy loss, ignoring index -1

    # Create DataLoader for batching images and labels
    dataloader = tagger.create_dataloader(images, labels)
    for epoch in range(epochs):
        epoch_loss = 0.0  # Track epoch loss for printing
        for t_images, t_labels in dataloader:
            optimizer.zero_grad()  # Reset gradients for the optimizer
            total_loss = 0  # Initialize total loss for batch

            # Encode images into feature embeddings using CLIP model
            image_features = tagger.clip_model.encode_image(t_images.to(device)).float()

            # Calculate similarity and loss for each category
            for category, embedding in model.items():
                embedding = embedding.to(device)  # Move embeddings to device
                # Calculate similarity between image features and text embeddings
                similarity = (image_features @ embedding.T).softmax(dim=-1)
                # Compute loss for the current category's similarity scores and labels
                loss = criterion(similarity, t_labels[category].to(device))
                total_loss += loss  # Accumulate loss for all categories
            
            
            # Backpropagate loss and update model parameters after each batch
            total_loss.backward()
            optimizer.step()

        # Print the loss at specified intervals
        if epoch % print_every == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss.item()}")


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
