<h1>Human detector</h1> <br>
Set of methods for detect human and draw bounding box around them. Based on YOLOv5s6.

<h3>How to run</h3>
Install requirements:
<code>pip install -r requirements.txt</code>.<br>
From project folder run app<br>
<code>python human_detector.py path\to-source\file.mp4 path\to-save\file.mp4</code><br>
You also can use specify the flag <code>-w</code> for indicate custom path to model weights <br>
<code>python human_detector.py path\to-source\file.mp4 path\to-save\file.mp4 -w path\to-my\file.pt</code><br>


<h3>Human detector docs</h3>
<code>human_detector/draw_bbox(image: np.array, bbox: np.array, label: str or None)</code><br>
Draw bounding box by coordinates in bbox parameter. <br>
<u>parameters:</u><br>
image: input image<br>
bbox: coordinates left top and bottom right corners of bounding box<br>
label: optional, text for drawing near bounding box. If label is None nothing be draw<br>
<u>return</u><br>
image with drawed bounding box<br>


<code>human_detector/draw_bbox_on_objects(image: np.array, table: pd.DataFrame, favorite: list or None)</code><br>
Draw all bounding boxes write in the table.<br>
<u>parameters:</u><br>
image: input image<br>
table: pandas DataFrame with next fields: <code>xmin, ymin, xmax, ymax</code> -- coordinates bounding boxes, 
<code>name</code> -- class name, <code>confidence</code> -- confidence recognize<br>
favorite: optional, list with specify name for drawing. If None then draws all classes from input table<br>
<u>return:</u><br>
image with drawed bounding boxes<br>


<code>human_detector/load_model(path_to_weights: str)</code><br>
Load YOLOv5 model from ultralytics with parameter <code>model</code> = "custom" and load weights from current device. 
<a href=https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#simple-example>more</a>.<br>
<u>parameter</u><br>
path_to_weights: path to weights on device, by default weights saved in folder 'models'<br>
<u>return</u><br>
YOLOv5 model like torch.nn.Module<br>


<code>human_detector/detect(path_to_video: str, path_to_save: str, path_to_weights: str)</code><br>
Detect humans on receive video and save new video with drawed bounding boxes by set path.<br>
<u>parameter</u><br>
path_to_video: path to video on device<br>
path_to_save: path for saving result video on device<br>
path_to_weights: path to weights model on device