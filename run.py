import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt

if not torch.cuda.is_available():
    print("CUDA is unavailable")
    exit()

device = torch.device('cuda')

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


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


model_id = "IDEA-Research/grounding-dino-tiny"
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Loaded processor and model!")

image = Image.open("./cyberpunk.jpg")

text = "a car. a person."


def preprocess_caption(caption: str) -> str:
    r = caption.lower().strip()
    if r.endswith("."):
        return r
    return r + "."


print("processing image and text...")
inputs = processor(images=image, text=text, return_tensors="pt")

for i, j in inputs.items():
    print("="*40)
    print(i)
    print()
    print(j)
    print("="*40)
    print()

inputs = inputs.to(device)

print("running through model...")
with torch.no_grad():
    outputs = model(**inputs)
print("got model output")

# process model outputs
width, height = image.size
processed_outputs = \
    processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[(height, width)],
        box_threshold=0.4,
        text_threshold=0.3,
    )

# for result in processed_outputs:
#     for k, v in result.items():
#         print(k)
#         print("\t", v)
#         print("\n")

results = processed_outputs[0]

plot_results(image, results['scores'], results['labels'], results['boxes'].tolist())
