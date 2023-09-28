# Importing necessary libraries for the inference script
import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImaged, AddChanneld, Orientationd, ScaleIntensityRanged, CropForegroundd, ToTensord
from monai.inferers import sliding_window_inference

# Assuming the 3D-UXNet model is correctly imported as UXNET.
# You may need to adjust the import statement based on your project structure.
from network_backbone import UXNET  

val_transforms = Compose(
    [
        AddChanneld(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ]
)

# Function to run inference and display the results
def inference(image_array=None, image_path=None, segmentation_path=None, roi_size=(96, 96, 96), sw_batch_size=2):

    if(image_path != None):
        original_nifti = nib.load(image_path)
        image = nib.load(image_path).get_fdata()
    elif image_array != None:
        image = image_array
    else:
        print("requires array or image path!")
        return None

    data = {"image": image}
    transformed_data = val_transforms(data)
    input_tensor = transformed_data["image"]
    input_batch = input_tensor.unsqueeze(0)

        
    model = UXNET(
        in_chans=1,
        out_chans=2,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )

    # Check if CUDA is available and move the model and input to GPU if possible
    if torch.cuda.is_available():
        print("CUDA")
        model.to('cuda:0')
        model.load_state_dict(torch.load('./best_metric_model_2500.pth'))
        input_batch = input_batch.to('cuda:0')
    elif torch.backends.mps.is_available():
        print("MPS")
        model.to('mps')
        model.load_state_dict(torch.load('./best_metric_model_2500.pth'), map_location=torch.device("mps"))
        input_batch = input_batch.to('mps')
    else:
        print("No GPU detected")   
        model.load_state_dict(torch.load('./best_metric_model_2500.pth'), map_location=torch.device("cpu"))

    model.eval()

    # 3. Run the model
    with torch.no_grad():
        output = sliding_window_inference(input_batch, roi_size, sw_batch_size, model)

    output = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    nifti_image = nib.Nifti1Image(output, original_nifti.affine, original_nifti.header)

    if segmentation_path != None:
        nib.save(nifti_image, "./label_" + segmentation_path)

    return output


image_path = '10018985_0000.nii.gz'

inference(image_path=image_path)
