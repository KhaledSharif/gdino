import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
import matplotlib.pyplot as plt

# 1. model id to run GDINO
gdino_model_id = "IDEA-Research/grounding-dino-tiny"

# 2. model id to run SAM
sam_model_id = "facebook/sam-vit-base"

# 3. load image from file
input_image = Image.open("./cyberpunk.jpg")

# 4. period separated list of objects to find
input_text = "a car. a person. stair steps."

if not torch.cuda.is_available():
    print("CUDA is unavailable")
    exit()

device = torch.device('cuda')

# Load SAM model
print("Loading SAM model...")
sam_model = AutoModelForMaskGeneration.from_pretrained(sam_model_id).to(device)
sam_processor = AutoProcessor.from_pretrained(sam_model_id)
print("SAM model loaded!")

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# Process the boxes from GDINO for SAM
def get_boxes_for_sam(results, height, width):
    boxes = []
    for box in results['boxes']:
        xmin, ymin, xmax, ymax = box
        xmin = float(xmin / width)
        ymin = float(ymin / height)
        xmax = float(xmax / width)
        ymax = float(ymax / height)
        boxes.append([[xmin, ymin, xmax, ymax]])
    return boxes


# Function to get SAM masks
def get_sam_masks(image, boxes):
    input_boxes = torch.tensor(boxes).unsqueeze(0)  # Add batch dimension
    inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = outputs.pred_masks.cpu().numpy()
    return masks


def plot_results(pil_img, scores, labels, boxes):
    print("labels:", labels)

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def preprocess_caption(caption: str) -> str:
    r = caption.lower().strip()
    if r.endswith("."):
        return r
    return r + "."


if __name__ == "__main__":
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(gdino_model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(device)
    print("Loaded processor and model!")

    print("processing image and text...")
    inputs = processor(images=input_image, text=input_text, return_tensors="pt")

    for i, j in inputs.items():
        print("=" * 40)
        print(i)
        print()
        print(j)
        print("=" * 40)
        print()

    inputs = inputs.to(device)

    print("running through model...")
    with torch.no_grad():
        outputs = model(**inputs)

    print("got model output, drawing boxes...")

    # process model outputs
    width, height = input_image.size
    processed_outputs = \
        processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[(height, width)],
            box_threshold=0.4,
            text_threshold=0.3,
        )

    results = processed_outputs[0]

    plot_results(input_image, results['scores'], results['labels'], results['boxes'].tolist())

    # Generate masks using SAM
    boxes_for_sam = get_boxes_for_sam(results, height, width)
    masks = get_sam_masks(input_image, boxes_for_sam)
    # masks.shape = (1, 4, 3, 256, 256)
    # np.min(masks, axis=(0, 2, 3, 4)) = array([-85.11738, -84.57804, -84.46066, -84.19439], dtype=float32)
    # np.max(masks, axis=()) = array([22.0906  , 22.039001, 21.984617, 21.810236], dtype=float32)

    # TODO: complete this code such that it draws the masks and displays the result

