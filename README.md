# **AI DEMO**

A simple AI demo model for helmet detection is designed to identify helmets in images using TensorFlow Lite. 

Trained on a custom dataset, optimized for real-time performance, and can be deployed on edge devices for safety and compliance monitoring in various environments.

## To execute the demo

To execute the demo run the below command in the terminal:

```shecll
python3 Detect_with_label.py --modeldir custom_model_lite/  --image helmet.jpg
```

<img src=".\result.jpg" alt="result" style="zoom: 47%;" />



**NOTE:**

Test Environment : ADLINK LEC-IMX95 

Yocto Version : 6.6.36 (scarthgap)