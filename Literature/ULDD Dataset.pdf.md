

{0}------------------------------------------------

# UL-DD: A Multimodal Drowsiness Dataset Using Video, Biometric Signals, and Behavioral Data

Morteza Bodaghi<sup>1</sup>, Majid Hosseini<sup>1</sup>, Raju Gottumukkala<sup>1\*</sup>,  
Ravi Teja Bhupatiraju<sup>1</sup>, Iftikhar Ahmad<sup>2</sup>, Moncef Gabbouj<sup>3</sup>

<sup>1</sup>University of Louisiana at Lafayette, USA.

<sup>2</sup>Tietoevry, Finland.

<sup>3</sup>Tampere University, Finland.

\*Corresponding author(s). E-mail(s): [raju.gottumukkala@louisiana.edu](mailto:raju.gottumukkala@louisiana.edu);

Contributing authors: [bodaghi.morteza@gmail.com](mailto:bodaghi.morteza@gmail.com);

[mjhouseiny@gmail.com](mailto:mjhouseiny@gmail.com); [ravi-teja.bhupatiraju@louisiana.edu](mailto:ravi-teja.bhupatiraju@louisiana.edu);

[iftikhar.ahmad@tietoevry.com](mailto:iftikhar.ahmad@tietoevry.com); [moncef.gabbouj@tuni.fi](mailto:moncef.gabbouj@tuni.fi);

## Abstract

In this study, we present a comprehensive public dataset for driver drowsiness detection, integrating multimodal signals of facial, behavioral, and biometric indicators. Our dataset includes 3D facial video using a depth camera, IR camera footage, posterior videos, and biometric signals such as heart rate, electrodermal activity, blood oxygen saturation, skin temperature, and accelerometer data. This data set provides grip sensor data from the steering wheel and telemetry data from the American truck simulator game to provide more information about drivers' behavior while they are alert and drowsy. Drowsiness levels were self-reported every four minutes using the Karolinska Sleepiness Scale (KSS). The simulation environment consists of three monitor setups, and the driving condition is completely like a car. Data were collected from 19 subjects (15 M, 4 F) in two conditions: when they were fully alert and when they exhibited signs of sleepiness. Unlike other datasets, our multimodal dataset has a continuous duration of 40 minutes for each data collection session per subject, contributing to a total length of 1,400 minutes, and we recorded gradual changes in the driver state rather than discrete alert/drowsy labels. This study aims to create a comprehensive multimodal dataset of driver drowsiness that captures a wider range of physiological, behavioral, and driving-related signals. The dataset will be available upon request to the corresponding author.

{1}------------------------------------------------

**Keywords:** multimodal driver drowsiness dataset, driver behavior analysis, continuous biosignals drowsiness detection, depth camera, biometric signals, steering behavior telemetry, early fatigue and drowsiness detection, open-access dataset

## 1 Introduction

Driver drowsiness is an important public safety concern, contributing to thousands of fatalities annually and elevates crash risk. Early drowsiness detection can prevent dangerous micro-sleep episodes before they cause accidents. Numerous vehicular drowsiness detection systems have emerged to address this public safety concern, and beginning in 2024, inclusion of early drowsiness system is mandated in all new vehicles within the European Union. The performance of these systems is dependent on the detection approach. Developing robust drowsiness detection systems requires high-quality, publicly available datasets.

Traditional detection systems have relied heavily on video-based monitoring of facial cues, such as blink frequency, eye closure duration, and yawning patterns [1–3]. These techniques tend to be too late for timely intervention and are prone to degradation under poor lighting conditions or due to inter-individual variability [4]. Physiological markers such as heart rate variability, skin temperature, and electroencephalogram (EEG) signals are alternative sensing technologies that enable early drowsiness detection [5, 6]. However, these signals have subject-specific variability [7, 8]. A multimodal approach that integrates facial expressions, physiological responses, and subtle behavioral patterns, such as steering micro-corrections, offers the opportunity to develop more robust methods for driver drowsiness detection.

Despite the increased interest in multimodal approaches, most of the existing public datasets primarily focus on single-modality signals (e.g., video or biometrics). This limits their practical use in advanced drowsiness detection, where multimodal data is crucial. To address this, we collected a multimodal dataset named the University of Louisiana Drowsiness Dataset (UL-DD) from a lab study with 19 different subjects with facial video (RGB, IR, and 3D Depth camera), biometric signals (Electrodermal Activity (EDA), Blood Volume Pulse (BVP), Inter-Beat Interval (IBI), Saturation of Peripheral Oxygen (SPO2), Respiration Rate (RR), HR, and ST), and behavioral signals (driver telemetry, posture video, grip pressure sensors, and accelerometer (ACC) data) to provide an open dataset for researchers for early drowsiness detection. This multimodal approach improves drowsiness detection by leveraging complementary information from physiological, vision, and behavioral data. For example, facial cues of drowsiness appear when the person is already drowsy; however, a physiological signal like HR can provide small signs of fatigue earlier, which can be combined with other data types for richer information [9].

We believe that this is a novel dataset that has the potential to allow researchers to develop metrics that may result in more early intervention by detecting latent and subtle declines in alertness for more timely interventions. Compared to other datasets, we collect a much more comprehensive set of streams from vision, biometry, and telemetry sensors.

{2}------------------------------------------------

The rest of the paper is organized as follows: Section 2 reviews related work, comparing UL-DD to existing public datasets. Section 3 describes the methodology, including participant recruitment, data collection protocols, and the collected multimodal signals. Section 4 details the data records, file structures, and Synchronization. Section 5 provided future opportunities and guidelines for multimodal analysis. Section 6 presents technical validation, including physiological and statistical analyses to confirm the dataset’s quality. Section 7 provides usage notes, while Section 8 outlines code availability. Finally, Section 9 discusses limitations and future work, followed by acknowledgments and references.

## 2 Related Work

**Table 1:** Drivers’ drowsiness detection publicly available datasets.

| Dataset  | Subjects         | Signals                                                                                                                                                                                                               | No of Classes |
|----------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| RLDD     | 60 (51 M, 9 F)   | 30 Hours of RGB Video                                                                                                                                                                                                 | 3             |
| NTHU-DDD | 36               | IR Video                                                                                                                                                                                                              | 5             |
| DROZY    | 14 (3 M, 11 F)   | EEG, EOG, EKG, EMG, and NIR                                                                                                                                                                                           | 8             |
| YawDD    | 107 (57 M, 50 F) | Video                                                                                                                                                                                                                 | 4             |
| DMD      | 37               | Video (capturing Face, Body, and Hand)                                                                                                                                                                                | 3             |
| 3MDAD    | 50 (38 M, 12 F)  | Video (RGB, IR, and Depth)                                                                                                                                                                                            | 2             |
| MRL Eye  | 37 (33 M, 4 F)   | 15000 infrared images                                                                                                                                                                                                 | 2             |
| NITYMED  | 21 (11 M, 10 F)  | Video                                                                                                                                                                                                                 | 5             |
| SEED-VIG | 23               | EEG and EOG signals (eye movement)                                                                                                                                                                                    | Score         |
| ULDD     | 19 (15 M, 4 F)   | Video (RGB, IR, Pose),<br>Grip pressure,<br>Telemetry ( Heading, Pitch, Roll, Speed, RPM, Gear),<br>Biometric Signals (HR, ST, ACC, EDA, BVP, IBI, SpO2)<br>Facial Landmarks<br>Facial Action Units<br>Pose Landmarks | 9             |

Monitoring driver state has been an active area of research over the past few years. In this section, we will review the publicly available datasets, their collection methods, and the types of signals captured. [10] collected 30 hours of RGB video from 60 subjects. Labels are alert, low vigilant, and drowsy. The video was captured by participants using their own mobile or web camera in different places. However, this dataset is limited to video signals with 10-minute durations for each subject, and it was not collected in a driving situation. NTHU-DDD [2] is a public dataset of IR videos from 36 people playing a driving simulation game in different situations, like normal driving, blinking slowly, yawning, falling asleep, and even laughing. However, it is based on subjects pretending to be drowsy, which might not be useful for detecting real drowsiness. DROZY [5] is a multimodal dataset of EEG, EOG, ECG, EMG, and near-infrared images (NIR) captured from 14 participants in a lab environment using the same camera to capture different levels of drowsiness. This dataset is limited to 30 minutes of data for each user, and it is labeled every 10 minutes, which does not

{3}------------------------------------------------

capture subtle changes in driver state. YawDD [11] includes videos recorded from two different angles in a parked car. 107 subjects participated in three situations: regular driving without talking, driving while talking or singing, and yawning while sitting in a parked car. NCKUDD [12] dataset has videos of 25 participants filmed in a parked car in both daylight and dark conditions. The videos cover different situations like regular driving, feeling sleepy, being distracted, talking, eating, talking on the phone while driving, and other unusual driving behaviors. SUST-DDD [13] dataset consists of 2074 videos, each 10 seconds from 19 participants, recorded by drivers’ phones during real driving, capturing moments of fatigue and normal driving. The dataset includes various lighting conditions and resolutions. It was labeled as drowsy or not drowsy based on watching the videos and voting. 3MDAD [14] used two cameras from frontal and side views in a real driving car to collect data from 60 people. The dataset provides RGB, infrared, and depth frames for each view. The MRL Eye [15] collected a total of 15,000 infrared images of the human eye, which are captured in various lighting conditions. The images are labeled open or closed, with the presence of glasses, reflections, gender, and subject ID. The NITYMED dataset [16] dataset is a public collection of videos of drivers in real cars moving under nighttime conditions. It contains 130 videos with different features and levels of drowsiness. The SEED-VIG[6] public dataset is a collection of EEG and EOG signals, as well as the corresponding vigilance level annotations, of subjects who played a driving game in a virtual driving system. The duration of the entire experiment was approximately 2 hours. DD-Pose [17] contains 330k measurements from multiple cameras acquired by an in-car setup during naturalistic drives. It also provides head pose annotations, steering wheel and vehicle motion information, and occlusion labels. DD-Pose is a diverse benchmark that can be used to develop and test head pose estimation methods for automotive applications. A vision-based public driver monitoring study[18] proposes a method for detecting and recognizing hand gestures and hand-object interactions using a camera pointed at a car dashboard. The study uses RGB and depth images captured by a Kinect sensor, including 7207 sample frames that can be used for normal driving and distraction analysis.

Some studies collected their private dataset capturing features of the eye, mouth, and BPM signal [19], video data of drivers in real driving road situations for eye tracking [20], RGB videos of participants in a parked car or simulator [21, 22], RGB, depth, and IR videos from 3 cameras capturing the face, body, and hands [23], thermal, NIR, visual, and physiological data [24], driving performance, eye movement, and EEG [25]. Table 1 summarizes publicly available datasets, the number of subjects, collected signals, and class labels.

Comparing UL-DD to multimodal datasets in broader fatigue and sleep research, rather than solely driver-specific studies, shows its versatile usage and value. For instance, [26] provides EEG and eye-tracking data from 23 subjects and classifies 3-level fatigue states, but lacks video or diverse biometrics. [27] offers wearable sensor signals for mental fatigue detection without real-time behavioral or video data. [28] presents a dataset with video and audio with a length of 40 hours for sentiment and fatigue analysis, yet misses depth video and physiological granularity. [29] focuses

{4}------------------------------------------------

![A photograph of the UL-DD driving simulator setup. It features a desk with three monitors displaying a driving simulation, a steering wheel, and pedals. A camera on a tripod is positioned to the right, and a white car seat is visible in the foreground.](bb08c83fc8939517c6803d65c69dd06b_img.jpg)

A photograph of the UL-DD driving simulator setup. It features a desk with three monitors displaying a driving simulation, a steering wheel, and pedals. A camera on a tripod is positioned to the right, and a white car seat is visible in the foreground.

**Fig. 1:** UL-DD driving simulator.

only on brain, muscle, and eye activity for driver behavior analysis missing actual behavioral signals like pose or telemetry data.

Unlike these, UL-DD stands out with three key contributions: 1- continuous 40-minute sessions that capture small and gradual changes in drowsiness; 2- a unique combination of multimodal signals (video, biometric, behavioral), and 3- detailed 9-level annotations to offer more precision in driver state. This positions the UL-DD as a unique resource, not only for driver monitoring but also for advancing broader fatigue and sleep studies.

## 3 Methods

We used American Truck Simulator as the driving simulator software. The software simulated trucking on U.S. highways, urban roads, and rural routes, with dynamic traffic and weather options. The simulator hardware included a three-monitor panoramic setup with a steering wheel and pedals. We did not use the manual gear system of the simulator. The subjects wore the following sensors: two wrist sensors and two on-the-wheel grip pressure sensors. They were observed by three cameras. The camera angles were depicted in Figure 2. 19 subjects, each participated in two 40-minute simulator sessions for a total of 80 minutes. Figures 1 and 3 show the driving simulator and sensors, respectively.

### 3.1 Participants

The participants were recruited primarily via flyers and recruitment emails in the engineering department of the University of Louisiana at Lafayette. While 30 subjects responded to the call, only 20 attended the study. Of these, 3 subjects only completed the first session and did not return to the second session. This left the study pool with 17 subjects who completed both sessions. We also included the 3 subjects who completed just the first session. One subject in the 2 session pool withdrew consent, leaving us with 16 subjects with 2 sessions and 3 subjects with 1 session, for a total of 19 subjects.

{5}------------------------------------------------

The age of the subject population ranged from 25 to 40 ( $M=27.3$ ,  $STD=2.3$ ). Other demographic information was not collected for privacy reasons. Inclusion criteria included individuals with driving experience, good general health, and no sleep disorders or cardiovascular conditions. The participants were asked to abstain from caffeine, tobacco, and other stimulants ten hours before the experiment. Further, participants were asked to avoid alcoholic beverages 24 hours before the study. Participation was voluntary and no incentives for participation were provided. Participants were informed that the cameras would collect their facial features and body posture during the tasks. The participants were also informed that the collected data would be released. The entire study protocol was supervised and approved by the UL Lafayette Institutional Review Board (IRB-22-038-IRI).

### 3.2 Video data

We used three different cameras during data collection to record the driver's face and movement. Figure 2 shows the direction of all three cameras in the simulator setup.

![Figure 2: Position and direction of different cameras in the simulator. (a) Direction of the infrared camera, showing a red cone of vision from the driver's seat. (b) Direction of the pose camera, showing a blue cone of vision from the steering wheel area. (c) Direction of the Zed 2 3D camera, showing a green and purple cone of vision from the side of the simulator.](0e62b4ac2303ba5f3ff10123a7c0f273_img.jpg)

(a) Direction of the infrared camera.
(b) Direction of the pose camera.
(c) Direction of the Zed 2 3D camera.

Figure 2: Position and direction of different cameras in the simulator. (a) Direction of the infrared camera, showing a red cone of vision from the driver's seat. (b) Direction of the pose camera, showing a blue cone of vision from the steering wheel area. (c) Direction of the Zed 2 3D camera, showing a green and purple cone of vision from the side of the simulator.

**Fig. 2:** Position and direction of different cameras in the simulator.

#### 3.2.1 ZED 2 Depth Camera

In this data collection, a "Zed 2" 3D depth camera (Figure 3b) was used for a special perception and environmental understanding. For each subject, the recorded video size is approximately 40 GB, with a resolution of 1344x376 at 60 frames per second, saved in MP4 format. The 3D camera was placed behind the steering wheel and next to the monitors to capture depth information without letting the steering wheel block the recording of facial data. We also shifted the camera and monitor positions to make sure the 3D camera did not get in the way of the drivers seeing the screens clearly. Figure 2c illustrates the position and direction of the 3D camera. For public access, the video was split into two separate views (left and right), each resized to 440x370 resolution, which greatly reduced the file size and made it simpler to work with.

#### 3.2.2 Infra-Red (IR) Camera

A 1080p IR camera was used to capture facial video, shown in Figure 3a. This camera has a high resolution and the ability to perform well under various lighting conditions,

{6}------------------------------------------------

![Figure 3: Devices used for data collection. The figure shows six items arranged in two rows. Top row: (a) IR camera (Microsoft Kinect), (b) ZED 2 camera, (c) Pose camera (EMEET Webcam). Bottom row: (d) Empathica E4 wristband, (e) O2 Max wristband, (f) Pressure Sensor.](a68671ef36320b8515aa85d288d82cf1_img.jpg)

(a) IR camera.

(b) ZED 2 camera.

(c) Pose camera.

(d) Empathica E4 wristband.

(e) O2 Max wristband.

(f) Pressure Sensor.

Figure 3: Devices used for data collection. The figure shows six items arranged in two rows. Top row: (a) IR camera (Microsoft Kinect), (b) ZED 2 camera, (c) Pose camera (EMEET Webcam). Bottom row: (d) Empathica E4 wristband, (e) O2 Max wristband, (f) Pressure Sensor.

**Fig. 3:** Devices used for data collection.

especially in low light conditions, which is common in actual driving situations at night. For each participant, the recorded video file size is approximately 35 GB, with a resolution of 640x360 at 60 frames per second, saved in MP4 format. Figure 2a shows the position and direction of the infrared camera. To focus on the facial region and remove the background, the video data was later cropped to a resolution of 240x220, which significantly reduced the file size to improve storage efficiency and accessibility.

#### 3.2.3 RGB Posture Camera

An EMEET Basic FHD 1080P Webcam (Figure 3c) was used to capture posture data through RGB video to analyze the subjects' physical movement. This camera provides full HD resolution, giving sharp and reliable images of body posture. For each participant, the video file is about 38 GB in size, recorded at a resolution of 640x480 and 60 frames per second, and saved as an MP4 file. The position and direction of the posture camera are shown in Figure 2b. To cut out the background and save space, we trimmed the video to a resolution of 620x470, reducing the overall file size.

### 3.3 Facial Features

We extracted 68 facial landmarks (FL) from the IR video for each user individually. This included subjects who did not consent to publication of video but consented to publication of features. All subjects consented to publication of facial features. Additionally, Facial Action Units (FAUs) were derived from these landmarks to capture key facial expressions and movements. Example FL points are shown in Figure 4.

{7}------------------------------------------------

![Figure 4: A composite image showing 68 facial landmarks. On the left, a diagram of a human face with 68 red dots labeled with numbers 1 through 68, indicating the positions of the landmarks. On the right, a photograph of a man's face with the same 68 red dots overlaid, showing the landmarks in a real-world context.](c834b9abb4ddf70e5d10641f87d5ff5b_img.jpg)

Figure 4: A composite image showing 68 facial landmarks. On the left, a diagram of a human face with 68 red dots labeled with numbers 1 through 68, indicating the positions of the landmarks. On the right, a photograph of a man's face with the same 68 red dots overlaid, showing the landmarks in a real-world context.

**Fig. 4:** Participants' 68 facial landmarks; this manuscript does not contain any images of the participants. We used one of the author's images to show the 68 facial landmarks.

**Table 2:** Facial Action Units.

| No | Action Unit | Name                       | No | Action Unit | Name          |
|----|-------------|----------------------------|----|-------------|---------------|
| 1  | AU1         | Inner Brow Raiser          | 16 | AU18        | Lip Puckerer  |
| 2  | AU2         | Outer Brow Raiser          | 17 | AU20        | Lip Stretcher |
| 3  | AU4         | Brow Lowerer               | 18 | AU22        | Lip Funneler  |
| 4  | AU5         | Upper Lid Raiser           | 19 | AU23        | Lip Tightener |
| 5  | AU6         | Cheek Raiser               | 20 | AU24        | Lip Pressor   |
| 6  | AU7         | Lid Tightener              | 21 | AU25        | Lips Part     |
| 7  | AU9         | Nose Wrinkler              | 22 | AU26        | Jaw Drop      |
| 8  | AU10        | Upper Lip Raiser           | 23 | AU27        | Mouth stretch |
| 9  | AU11        | Nasolabial Furrow Deepener | 24 | AU28        | Lip Suck      |
| 10 | AU12        | Lip Corner Puller          | 25 | AU41        | Lid droop     |
| 11 | AU13        | Cheek Puffer               | 26 | AU42        | Slit          |
| 12 | AU14        | Dimpler                    | 27 | AU43        | Eyes Closed   |
| 13 | AU15        | Lip Corner Depressor       | 28 | AU44        | Squint        |
| 14 | AU16        | Lower Lip Depressor        | 29 | AU45        | Blink         |
| 15 | AU17        | Chin Raiser                | 30 | AU46        | Wink          |

Facial Action Coding System (FACS) is a comprehensive framework for understanding facial expressions, particularly the discrete movements and gestures referred to as Action Units (AUs). Recent studies showed that using FAUs can reduce the amount of unwanted information from the video data, reduce the data size and computational cost, and increase the model performance by removing redundant information (noise) like skin tone, lighting, and background [30]. We generated and provided a set of action units along with the other data types, as listed in Table 2. However, we mostly focused on upper-face facial action units and yawing.

{8}------------------------------------------------

### 3.4 Biometric Signals

We used two wristbands, the Empatica E4 (Figure 3d) and Checkme O2 Max (Figure 3e), for collecting biometric signals. The following biometric signals are provided in the dataset:

#### 3.4.1 Multi-sensor Wearable

Below are the signals that were collected using Empatica E4:

- **HR:** Heart Rate typically ranges from 60 to 100 beats per minute when a person is resting. However, this rate can change depending on environmental factors like physical activity, mood, and health condition. Wearable sensors track these changes and patterns, which may potentially offer insights about drowsiness states[31].
- **TEMP:** Skin Temperature, another signal from the E4, serves as an indicator of peripheral vasodilation and has been associated with fluctuations in alertness and arousal levels[32]. Wearable sensors capture small body temperature changes, which usually happen during sleep [33]. This can be studied with other physiological signals to detect signs of fatigue and drowsiness [34].
- **EDA:** Also known as Galvanic Skin Response (GSR), measures sweat gland activity and gives information about emotional arousal and stress. Fluctuations in EDA can show physiological states that are related to drowsiness, such as low alertness and fatigue [35].
- **ACC:** Accelerometer data, measured by sensors, records object acceleration. These sensors have diverse applications, such as human action recognition and step counting. Wearable ACCs track movement patterns, offering information about activity levels[36].
- **BVP:** Blood Volume Pulse measures blood flow changes to assess cardiovascular activity. Variations in BVP can indicate physiological changes related to drowsiness [37].
- **IBI:** InterBeat Interval represents the time between heartbeats. Changes in IBI patterns can reflect the autonomic nervous system and heart activity that may help detect early signs of decreased vigilance [38].

#### 3.4.2 Pulse Oximeter

Below are the signals that were collected using Checkme O2 Max Pulse Oximeter:

- **Pulse Rate:** It measures the number of heartbeats per minute and is identical to HR. However, since it is measured using different devices, we refer to them separately to avoid confusion.
- **Motion:** Motion data, collected alongside SpO2, to help identify and flag potential motion artifacts in the SpO2 readings [39].
- **SpO2:** Saturation of peripheral oxygen measures blood oxygen levels and helps evaluate overall physical health. Drops or changes in SpO2 can be tied to uneven breathing patterns that happen during drowsiness, making it a helpful sign for drowsiness monitoring [40].

{9}------------------------------------------------

### 3.5 Behavioral Data

Behavioral data provides complementary information for biometric signals and facial data to detect drowsiness. We collect behavioral data to monitor cues such as posture, head tilting, grip pressure changes, and different driving patterns, which can be fatigue indicators [41]. We collected three types of behavioral data:

- **Pose:** We captured posture data using a webcam, as explained in 3.2.3. We extracted pose landmarks (PL) from the video using MediaPipe Pose [42], a real-time pose estimation framework. Pose changes such as slouching, head tilting, or leaning to one side may show fatigue or loss of focus during long driving sessions [43].
- **Grip pressure:** Grip pressure data was collected using a pressure sensor installed on both sides of the steering wheel, shown in Figure 3f. Changes in grip strength or asymmetry in pressure distribution can indicate drowsiness-related muscle relaxation or changes in driver engagement. Consistent pressure patterns may reflect alertness, while drops or fluctuations could correspond to drowsiness [44].
- **Telemetry:** Telemetry data was collected from the driving simulator software and offers more insights into driving patterns, such as speed and lane changing. Inconsistent speed or delayed reactions may indicate drowsiness [45].

### 3.6 Data collection protocol

Our data collection took place in a controlled laboratory setting, where we used the American Truck Simulator software deployed in a custom-built driving simulator. We used three cameras for recording facial and pose data (see Section 3.2). We used two different wristbands to collect biometric signals (see Section 3.4). Additionally, other behavioral data like telemetry and grip pressure were collected through the game and sensors installed on the steering wheel, respectively.

**Table 3:** Description of the drowsiness data collection tasks and durations.

| No.           | Task        | Action performed                                          | Time   |
|---------------|-------------|-----------------------------------------------------------|--------|
| 1             | Preparation | Familiarize the driver with the simulator                 | 10 min |
| 2             | Rest        | Sitting behind the wheel and relax before data collection | 5 min  |
| 3             | Driving     | Driving in a simulated space and staring data collection  | 40 min |
| 4             | Survey      | Complete KSS questionnaire                                | 5 min  |
| <b>Total:</b> |             |                                                           | 60 min |

The data was collected from 19 subjects (15 males, 4 females) performing driving tasks shown in Table 3 at different drowsiness levels. We asked them to attend the driving sessions once they were alert and showed no signs of drowsiness, and once when they felt drowsy. Drowsiness was verified using the Karolinska Sleepiness Scale questionnaire, requiring a score of  $\geq 6$  (some signs of sleepiness) to start. For each session, the aim was to complete an assigned route; much of the route was on a simulated major limited highway. At first, we asked them to drive for 10 minutes to get familiar with the simulator, followed by a 5-minute rest, and then they drove for 40 minutes,

{10}------------------------------------------------

approximately 40 to 50 virtual miles, depending on speed and traffic conditions. Participants followed the simulator’s GPS navigation to avoid other distractions beyond speed control and response to traffic. During driving, they were asked to report their drowsiness level based on the KSS questionnaires every four minutes. The labeled data of the active driving session is made available in the dataset.

### 3.7 Drowsiness detection surveys

Each task was divided into ten four-minute sub-intervals, totaling 40 minutes for each session. The KSS was chosen because it has a standard 9-level scoring to capture gradual changes in drowsiness, which is widely used for driving studies [46]. Table 4 shows the KSS questionnaires that were asked to evaluate participants’ drowsiness levels from 1 to 9 for each sub-interval. To ensure the reliability of these scores, we calculated Cronbach’s alpha, a standard measure of reliability [47]. The resulting  $\alpha$  coefficient was 0.99, indicating excellent internal consistency. This means the repeated scores reliably captured how each participant’s drowsiness changed over time.

**Table 4:** Karolinska sleepiness scale.

| Score | Description                                             |
|-------|---------------------------------------------------------|
| 1     | Extremely alert                                         |
| 2     | Very alert                                              |
| 3     | Alert                                                   |
| 4     | Rather alert                                            |
| 5     | Neither alert nor sleepy                                |
| 6     | Some signs of sleepiness                                |
| 7     | Sleepy, but no effort to keep awake                     |
| 8     | Sleepy, some effort to keep awake                       |
| 9     | Very sleepy, great effort to keep awake, fighting sleep |

## 4 Data records

The UL-DD dataset includes three main folders of Video\_Data, CSV\_Files, and Extracted\_Features, along with a separate Labels.CSV file. Each Folder has 19 folders from A to S, which show the participant’s ID. For each participant, there are two folders, A and D, which stand for the Awake and Drowsy sessions, respectively. File names begin with the participant’s ID, followed by the signal name, and the session. For instance, *A\_IR\_D* shows IR video data for participant A, Drowsy session, whereas *B\_FL\_A* refers to FLs extracted from video data for participant B and awake session. The labels are also available for each participant separately. We labeled the data every four minutes, and the label files contained ten instances. A detailed description of the files is included in the following section.

### 4.1 Data files description

The following is the description of files in the task sub-folders:

{11}------------------------------------------------

- **User\_HR\_Session.csv:** The file contains a single column that corresponds to the average heart rate. The HR signal was recorded at a frequency of 1 Hz.
- **User\_ACC\_Session.csv:** The file contains three columns corresponding to the x-axis, y-axis, and z-axis accelerometer data, respectively. The 3-axis ACC data was recorded at a frequency of 32 Hz.
- **User\_EDA\_Session.csv:** The file contains a single column that includes average electrodermal activity. The EDA signal was recorded at a frequency of 4 Hz.
- **User\_TEMP\_Session.csv:** The file is a single-column CSV format that contains the average skin temperature. The TEMP signal was recorded at a frequency of 4 Hz.
- **User\_BVP\_Session.csv:** The file is a single-column CSV format that contains the blood volume pulse signal. The TEMP signal was recorded at a frequency of 4 Hz.
- **User\_IBI\_Session.csv:** The file contains two columns: the first column shows the time stamp, and the second column contains interbeat interval values, which are the difference between two consecutive rows of the first column. This signal contains time, which means IBI can not be interpreted in Hz frequency.
- **User\_O2M\_Session.csv:** The file contains three columns: blood oxygen level, pulse rate, and motion activity data. The SPO2 signal was recorded at a frequency of 0.5 Hz.
- **User\_RGP\_Session.csv & User\_LGB\_Session.csv :** Each file contains a single column that includes the grip pressure data on the steering wheel for the right and left hands. The GP data was recorded at a frequency of 3 Hz.
- **User\_Telemetry\_Session.csv :** The file contains ten columns: Timestamp, raw rendering timestamp, raw simulation timestamp, raw paused simulation timestamp, pitch, roll, speed, rpm, and gear. This data captures the driving behavior from the American Truck Simulator. The telemetry signals were recorded at a frequency of 60 Hz.
- **User\_FL\_Session.csv:** The file contains  $68 \times 2 = 136$  columns representing the x and y coordinates of the participant's 2D FLs. These 68 landmarks were extracted from the "IR.mp4" video using the Dlib [48] library. The data was sampled at a frequency of 60 fps, with the corresponding frame number recorded in the first column.
- **User\_PL\_Session.csv:** The file contains  $33 \times 3 = 99$  columns that present the x, y, and z coordinates of the participant's PLs. These PLs were extracted from the "Pose.mp4" video files using MediaPipe Pose[42]. The data was sampled at a frequency of 60 fps, with the corresponding frame number recorded in the first column.
- **User\_FAU\_Session.csv:** The file contains thirty columns that present the main thirty facial action units explained in section 3.3. FAUs were extracted from the "FL.csv" files. The data was sampled at a frequency of 60 fps, with the corresponding frame number recorded in the first column.
- **User\_IR\_Session.mp4:** The video file contains raw trimmed infrared video files of participants who consented to release their videos. This file is not available for all participants. The video has a frequency of 60 fps.

{12}------------------------------------------------

- **User\_R3D\_Session.mp4 & User\_L3D\_Session.mp4:** These video files contain the right and left angles of raw trimmed depth camera recordings for participants who consented to release their videos. These files are not available for all participants, but the corresponding FLs are provided for all the subjects. The video was recorded at 60 fps.
- **User\_Pose\_Session.mp4:** This raw, trimmed video files capture a side view of the driver for participants who consented to release their videos. It is not available for all participants, but corresponding PLs are provided for all the subjects. The video was recorded at 60 fps.
- **Labels.csv:** The file contains ten columns, each representing the driver’s drowsiness level assessed over a four-minute interval with the corresponding User\_Session information in the first column.

### 4.2 Synchronization

In multimodal learning, aligning data over time is essential to properly combine different data streams for successful model training and analysis. Data often comes from various sources, each with its own recording frequency, delay, and time reference, which can cause misalignment if not appropriately handled. Without good synchronization, timing differences can lead to mistakes, weaken model accuracy, and distort how modalities relate to each other [49]. There are several approaches to synchronization, depending on whether a shared reference is available. The best scenario would be using a single clock for all devices, but this isn’t always practical. A more feasible approach is timestamping, where each data point gets an exact time label, allowing it to line up data based on when it was actually recorded. This method ensures temporal consistency across different modalities, even when their sampling rates vary [50].

For our dataset, we synchronized all modalities using manually recorded session start and end times as reference points. Video data captured simultaneously from three cameras was trimmed to match the session window. Biometric signals from the E4 wristband were also synchronized using epoch timestamps, ensuring alignment with the session timeline. Similarly, physiological data collected from the Checkme O2 Max wristband, including SPO2, pulse rate, and motion, contained actual clock timestamps, which facilitated direct alignment with the session times. Grip pressure data was synchronized by matching the recorded start time in its file to the manually recorded session start. Telemetry data, which began when the driving simulation started, was aligned by observing its initiation in the video recordings and trimming excess data accordingly.

In cases where interruptions occurred during data collection, such as software crashes, we relied on notes taken during the sessions and video recordings to identify and correct the affected parts. In this way, all modalities remained consistently aligned despite these interruptions.

{13}------------------------------------------------

![A flowchart illustrating a multimodal analysis pipeline. The process is divided into five stages: Data Collection, Individual Pre-processing, Feature Extraction, Multimodal Pre-processing, and Fusion Strategy. In the Data Collection stage, three main data types are gathered: Biometric Signals (HR, BVP, IBI, EDA, TEMP, ACC), Facial Data (IR Video, 3D depth Video), and Behavioral Data (Pose Video, Grip Pressure, Telemetry). In the Individual Pre-processing stage, each data type undergoes specific cleaning and normalization steps: Biometric signals are filtered, outliers are removed, and smoothed; Facial and Behavioral data are normalized for noise, illumination, and ROI. In the Feature Extraction stage, features are derived from the pre-processed data: Biometric signals yield statistical features in both time and frequency domains; Facial data yields facial landmarks, action units, and a 3D depth map; Behavioral data yields pose landmarks and body movement features. In the Multimodal Pre-processing stage, these features are aligned temporally, resampled, and normalized. Finally, in the Fusion Strategy stage, a Multimodal Fusion Algorithm combines the features for Classification.](4801720824e4b5e2361a5564f91cfb70_img.jpg)

A flowchart illustrating a multimodal analysis pipeline. The process is divided into five stages: Data Collection, Individual Pre-processing, Feature Extraction, Multimodal Pre-processing, and Fusion Strategy. In the Data Collection stage, three main data types are gathered: Biometric Signals (HR, BVP, IBI, EDA, TEMP, ACC), Facial Data (IR Video, 3D depth Video), and Behavioral Data (Pose Video, Grip Pressure, Telemetry). In the Individual Pre-processing stage, each data type undergoes specific cleaning and normalization steps: Biometric signals are filtered, outliers are removed, and smoothed; Facial and Behavioral data are normalized for noise, illumination, and ROI. In the Feature Extraction stage, features are derived from the pre-processed data: Biometric signals yield statistical features in both time and frequency domains; Facial data yields facial landmarks, action units, and a 3D depth map; Behavioral data yields pose landmarks and body movement features. In the Multimodal Pre-processing stage, these features are aligned temporally, resampled, and normalized. Finally, in the Fusion Strategy stage, a Multimodal Fusion Algorithm combines the features for Classification.

Fig. 5: Multimodal analysis pipeline.

## 5 Future Opportunities

### 5.1 Multimodal Analysis Guidelines

In this study, we used a pipeline architecture to allow for multimodal analysis, as shown in Figure 5. As previously mentioned, we collected three distinct modalities: biometric, facial, and behavioral data. Several feature extraction methods and pre-processing steps can be done based on the modality type and the task that we are dealing with. Possible suggestions are provided in the following sections:

#### 5.1.1 Feature Extraction

Feature extraction plays an important role in drowsiness modeling by adding meaningful information to raw data. The features that can be derived from different modalities are explained below:

- **Biometric Signals:** For signals like HR, IBI, BVP, EDA, TEMP, SpO2, and pulse rate, we can extract several time-domain features, such as the mean, standard deviation, and root mean square. These features help us understand the body's normal state and how it changes over time [51]. We can also extract frequency-domain features, such as power spectral density from HRV and frequency analysis of EDA signals [52] to help identify phasic (rapid) and tonic (baseline) components of arousal changes [53]. Additionally, data like ACC and motion sensors offer insight into physical activity levels. Features like mean acceleration magnitude, entropy, and movement frequency are useful for detecting reduced movement. Other statistical metrics, such as skewness, kurtosis, and signal entropy, help capture small and large changes over time [54].

{14}------------------------------------------------

- **Facial Data:** Facial video data provides both visual and depth-based information for identifying signs of drowsiness. From these videos, FLs and FAUs are the most common features that can be extracted using DLib [48] or MediaPipe [42]. The key points are those related to the eyes, mouth, and eyebrows. Commonly used features include the Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink frequency, mouth opening angle, and the percentage of eye closure over time (PER-CLOS). These metrics are well-known indicators of drowsiness [55]. We also track head movements, such as nodding or tilting. Depth maps help us to extract spatio-temporal features, such as frame-to-frame changes in landmark positions. These can help capture tiny movements of the eyelid or small changes in posture, which are often early signs of fatigue.
- **Behavioral Data:** Behavioral data captures the physical movement and interaction of the driver with the vehicle, which is affected by drowsiness. Potential features include extracting PLs from video using tools like OpenPose[56] and MediaPipe. Features, such as posture angles, movement amplitude, and stillness duration, can be extracted with the focus on key points like shoulders, spine, and head. For grip pressure and telemetry data, time-domain, frequency-domain, and statistical features are mostly useful. For example, average grip pressure, grip duration, speed variability, and engine RPM patterns can provide valuable information about the driver’s state [57].

#### 5.1.2 Data Pre-processing

We do pre-processing to make sure data from different modalities are cleaned, aligned and transformed into suitable formats for fusion and classification. In the public version of the dataset, each modality has a unique sampling frequency, as explained in Section 4.1. Here are some suggested pre-processing steps for each modality:

- **Biometric Signals:** These signals can easily become noisy from sudden movements, sensor placement, or environmental factors like temperature and humidity. To clean them, techniques like band-pass filtering are suitable to remove high-frequency noise [58]. At the same time, a method called independent component analysis (ICA) can help separate motion artifacts from signals like HR, BVP, EDA, and ACC data [59]. To make sure everything is consistent across different subjects, we can normalize the data for each participant separately. Also, resampling or interpolation techniques can address different sampling rate issues so they line up properly with each other.
- **Facial Data:** Facial video data require pre-processing to address differences in lighting, occlusions, and camera-specific noise. Standardizing video frames to a fixed resolution facilitates consistent feature extraction. Histogram equalization[60] or adaptive gamma correction[61] can mitigate lighting variations, while median or Gaussian filters help remove depth and pixel-level noise [62]. Additionally, optical flow-based interpolation can reconstruct missing frames to maintain temporal consistency in video sequences.
- **Behavioral Data:** Behavioral data, including video and time-series data, require similar pre-processing techniques as those mentioned for biometric and facial data.

{15}------------------------------------------------

Filtering techniques are useful for reducing sensor or environmental noise, while cropping regions of interest ensure that only relevant areas are analyzed. Normalization and resampling methods standardize data across participants and align different sampling rates. Additionally, outlier removal techniques help eliminate anomalies.

- **Multimodal:** Before integrating modalities into a multimodal framework, additional pre-processing is needed for synchronization, alignment, and compatibility. Temporal alignment means adjusting all modalities to a single sampling rate, such as 4 Hz, to make synchronized data fusion easier. windowing methods can be used to segment the data into fixed-length time windows for analysis. Normalization can be helpful in consistency across modalities and prevents bias due to scale differences. Finally, methods like Principal Component Analysis (PCA) and manifold learning can be used to decrease the data dimension, which reduces the computing cost, particularly when working with data that has many features[53].

#### 5.1.3 Integration Across Modalities

After pre-processing, several fusion strategies can be applied depending on the task:

- **Early-Fusion:** This technique merges raw or basic features from all modalities into one feature vector before performing classification. It helps identify connections between modalities, but it can lead to challenges with higher data dimensions and noise (see Figure 6a).
- **Late-Fusion:** In this method, each modality is processed independently to provide separate classification, and then these individual decisions are combined using techniques like majority voting, basic averaging, or weighted combination. This method is stronger against noise for each modality, and it is more flexible in data pre-processing, but it might miss relationships between modalities (see Figure 6b).
- **Intermediate-Fusion:** This method combines modalities at an intermediate stage after extracting higher-level features and representations but before the final classification. This allows a balance for capturing correlations between modalities while maintaining flexibility in feature representation. It has a higher computational cost and better performance in most cases compared to early and late fusion approaches (see Figure 6c).

![Diagram illustrating three conventional methods for multimodal data fusion: (a) Early-fusion, (b) Late-fusion, and (c) Intermediate-fusion.](a83ba9e3e2c1e21dd69953a7b09e45b4_img.jpg)

The diagram shows three flowcharts for multimodal data fusion:

- (a) **Early-fusion:** Three input boxes labeled 'Modality 1', 'Modality 2', and 'Modality 3' all point to a single red diamond labeled 'Fusion'. This diamond points to a box labeled 'Classifier', which then points to an oval labeled 'Result'.
- (b) **Late-fusion:** Three input boxes labeled 'Modality 1', 'Modality 2', and 'Modality 3' each point to a box labeled 'Classifier 1', 'Classifier 2', and 'Classifier 3' respectively. Each classifier points to a red diamond labeled 'Fusion'. This diamond points to an oval labeled 'Result'.
- (c) **Intermediate-fusion:** Three input boxes labeled 'Modality 1', 'Modality 2', and 'Modality 3' each point to a box labeled 'Classifier 1', 'Classifier 2', and 'Classifier 3' respectively. Each classifier points to a green hexagon labeled 'Representation 1', 'Representation 2', and 'Representation 3' respectively. These three hexagons point to a red diamond labeled 'Fusion'. This diamond points to a box labeled 'Classifier 4', which then points to an oval labeled 'Result'.

Diagram illustrating three conventional methods for multimodal data fusion: (a) Early-fusion, (b) Late-fusion, and (c) Intermediate-fusion.

**Fig. 6:** Conventional methods for multimodal data fusion.

{16}------------------------------------------------

## 6 Technical validation

In this section, we provide a quality validation for the various captured signals to determine their suitability for driver drowsiness detection and to examine the correlation between the signals and different drowsiness levels. To achieve this, we applied statistical tests to assess the significance of the signals in relation to drowsiness levels, considering both direct and inverse proportionality, as well as individual variability. We used signal-to-noise ratio (SNR) analysis to evaluate signal quality. Label consistency was verified using Cohen’s Kappa for inter-rater agreement. We have also tested the multimodal aspect of the data by performing classification tasks using two well-known machine learning models with k-fold cross-validation.

### 6.1 Physiological Validation

To validate the collected biometric signals and how they change with different levels of drowsiness, we present box plots (Figure 7) that show the distribution of each signal across all participants. These plots are divided into three categories of drowsiness: low, medium, and high. For illustration, we removed the outliers for each signal per subject and then standardized them to have a mean of zero and a standard deviation of one to be consistent for all subjects. These distributions help validate the reliability of the collected signals and highlight patterns of change in response to increasing drowsiness levels. These plots show the inherent inter-subject variability in physiological data that is necessary for understanding individual differences in response to drowsiness.

![](fdcfba1180dc160c7d539c5fb2a6c1e6_img.jpg)

- (a) Pulse Rate (PR)**: Title: *Boxplot of Pulse\_Rate by Subject and Drowsiness Level*; X-axis: *Subject* (Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S); Y-axis: *Pulse\_Rate*; Legend: *Drowsiness Level* (Low, Med, High).
- (b) Electrodermal activity (EDA)**: Title: *Boxplot of eda by Subject and Drowsiness Level*; X-axis: *Subject* (Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S); Y-axis: *eda*; Legend: *Drowsiness Level* (Low, Med, High).
- (c) Skin temperature (TEMP)**: Title: *Boxplot of temp by Subject and Drowsiness Level*; X-axis: *Subject* (Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S); Y-axis: *temp*; Legend: *Drowsiness Level* (Low, Med, High).
- (d) Blood oxygen saturation (SPO2)**: Title: *Boxplot of SPO2 by Subject and Drowsiness Level*; X-axis: *Subject* (Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S); Y-axis: *SPO2*; Legend: *Drowsiness Level* (Low, Med, High).

**Fig. 7:** Box plots of biometric signals for all the subjects and three levels of drowsiness.

{17}------------------------------------------------

#### 6.1.1 Statistical Analysis

To validate the physiological data, we first tested whether it followed a normal distribution or not, using the Shapiro-Wilk[63] test. The results indicated that the data was not normally distributed. Because of this, we used the Kruskal-Wallis Test [64], which is a non-parametric option instead of ANOVA. This test looks at the rankings of different groups to see if the differences in how the signals are spread across drowsiness levels are meaningful or just random chance. It assesses whether the samples come from the same distribution or exhibit fundamental differences. Since the number of drowsiness level labels is limited (KSS=9), we categorized drowsiness levels into Low, Medium, and High, as outlined below:

- High: Drowsiness levels greater than 6
- Medium: Drowsiness levels between 4 and 6
- Low: Drowsiness levels less than 4

All tests show significant differences between 'Low' and 'Medium' drowsiness levels with P-values less than 0.001, so that we can reject the null hypothesis, except for BVP with a p-value of 0.059. Similarly, all measurements exhibit significant differences between 'Medium' and 'High' drowsiness levels except ACC\_X, with a p-value of 0.098. Finally, all measurements show significant differences between 'Low' and 'High' drowsiness levels for all biometric signals except BVP, with the p-value of 0.055.

We also assessed the statistical significance of the data using the Linear mixed effects model[65] that extends traditional linear regression by incorporating both fixed and random effects. Fixed effects capture the overall population-level relationships between independent variables and the dependent variable, while random effects account for variations at the group or individual level. This dual structure allows for more accurate data modeling with hierarchical or clustered arrangements, such as repeated measurements from the same subjects or data grouped by categories. We used the same binning for the drowsiness level for linear models with mixed effects. We explored the relationship between various physiological parameters and levels of drowsiness by incorporating inter-subject variability among 19 participants. The linear model with mixed effects shows that the BVP signals do not correlate significantly with drowsiness, so we can not reject the null hypothesis. The remaining biometric signals showed a significant correlation ( $P < 0.001$ ), with one exception in HR when comparing medium and high drowsiness levels with a p-value of 0.2. As we used two different sensors for capturing heart-related activity, PR shows more consistent results across drowsiness levels, so we recommend relying on PR data for analysis.

Based on the test results presented in Table 5, the following observation was made:

- **EDA** decreased with increased drowsiness levels, indicating heightened sympathetic nervous system activity when individuals are more drowsy. A group variance of 0.727 suggests small variability in EDA levels across subjects. The z-score values indicate highly significant effects, meaning these coefficients differ significantly from zero. The residual standard deviation indicates a small amount of unexplained variability in the data, but the model's fit remains reasonably accurate.

{18}------------------------------------------------

**Table 5:** Mixed-Effects Model Results for Biometric Signals.

| Signal      | Coef.     | Est.   | Std.Err | Z-score  | P-value | Group Var | Residual Std |
|-------------|-----------|--------|---------|----------|---------|-----------|--------------|
| <b>EDA</b>  | Intercept | 0.565  | 0.196   | 2.888    | 0.004   | 0.727     | 0.510        |
|             | T.Low     | 0.220  | 0.002   | 90.490   | <0.001  |           |              |
|             | T.Med     | 0.075  | 0.002   | 33.452   | <0.001  |           |              |
| <b>TEMP</b> | Intercept | 33.099 | 0.264   | 125.508  | <0.001  | 1.321     | 1.085        |
|             | T.Low     | -1.120 | 0.005   | -216.406 | <0.001  |           |              |
|             | T.Med     | -1.397 | 0.005   | -293.733 | <0.001  |           |              |
| <b>HR</b>   | Intercept | 74.303 | 1.688   | 44.025   | <0.001  | 54.113    | 6.567        |
|             | T.Low     | 1.983  | 0.031   | 63.328   | <0.001  |           |              |
|             | T.Med     | -0.037 | 0.029   | -1.272   | 0.2     |           |              |
| <b>BVP</b>  | Intercept | 0.037  | 0.669   | 0.055    | 0.956   | 8.043     | 51.520       |
|             | T.Low     | -0.025 | 0.245   | -0.101   | 0.920   |           |              |
|             | T.Med     | -0.030 | 0.225   | -0.132   | 0.895   |           |              |
| <b>SPO2</b> | Intercept | 94.946 | 0.255   | 372.445  | <0.001  | 1.234     | 1.674        |
|             | T.Low     | 0.745  | 0.008   | 93.340   | <0.001  |           |              |
|             | T.Med     | 0.851  | 0.007   | 116.014  | <0.001  |           |              |
| <b>PR</b>   | Intercept | 74.981 | 1.954   | 38.366   | <0.001  | 72.564    | 5.827        |
|             | T.Low     | 2.733  | 0.028   | 98.388   | <0.001  |           |              |
|             | T.Med     | 0.758  | 0.026   | 29.698   | <0.001  |           |              |

- **TEMP** increased with increasing drowsiness levels, reflecting a growth in skin temperature as individuals become more drowsy, as the body tends to warm up in Peripheral areas [66]. This occurs due to increased blood flow during the transition to sleep, particularly in the fingers and hands, where wearables typically measure temperature[67, 68]. A group variance of 1.321 indicates moderate variability in temperature across subjects. The z-scores are extremely large and significantly different from zero. The residual standard deviation suggests that while the model fits the data well, there remains a moderate amount of unexplained variability in the temperature measurements.
- **SPO2** exhibited slight increases at lower drowsiness levels, which may reflect improved respiratory or cardiovascular function associated with alertness. A small group variance indicates small variability across subjects. The exceptionally large z-scores and their high significance underline the robustness of these associations.
- **PR** increased with lower drowsiness levels. This suggests heightened cardiovascular activity when individuals are more alert. The large group variance indicates substantial variability in pulse rate across subjects, reflecting individual differences in heart rate responses. The residual standard deviation points to moderate unexplained variability in pulse rate, suggesting that other factors beyond drowsiness levels may also influence these measurements.

### 6.2 Signal Quality Assessment

To make sure that our dataset is reliable, we evaluated signal quality for biometric signals, behavioral data, and video data. Firstly, for biometric signals (HR, EDA, TEMP, SpO2, pulse rate, ACC) and behavioral data (grip pressure), we conduct a Signal-to-noise ratio (SNR) analysis using a 5-minute rest period. Specifically, for

{19}------------------------------------------------

each subject and session, we computed the SNR of the signals using a third-order Butterworth low-pass filter to separate the signal from noise. Then, the SNR was calculated as  $SNR = 10 \log_{10} \left( \frac{P_{signal}}{P_{noise}} \right)$ , where  $P_{signal}$  and  $P_{noise}$  are the power of the filtered signal and noise, respectively [69]. The noise was calculated by subtracting the filtered signal from the original. For HR, SPO2, and pulse rate, a cutoff frequency of 1.5 Hz was used to preserve cardiac-related components[70]. EDA was filtered at 1.5 Hz to retain both tonic and phasic components[71], and TEMP and ACC were filtered at 1.0 Hz due to their slower-varying nature. The grip pressure was also set to 1.0 Hz to minimize high-frequency fluctuations [72]. The results are presented in table 6:

**Table 6:** SNR analysis results for biometric and behavioral signals.

| Signal               | Avg SNR (dB) | SD (dB) | Min-Max (dB)  | % <10 dB |
|----------------------|--------------|---------|---------------|----------|
| <b>HR</b>            | 45.67        | 3.19    | 38.48 - 51.53 | 0        |
| <b>EDA</b>           | 21.20        | 10.35   | 1.55 - 43.26  | 14.29    |
| <b>TEMP</b>          | 26.44        | 5.03    | 15.56 - 36.13 | 0        |
| <b>SPO2</b>          | 30.23        | 3.83    | 23.12 - 37.93 | 0        |
| <b>Pulse Rate</b>    | 31.95        | 3.91    | 25.96 - 40.53 | 0        |
| <b>ACC</b>           | 19.59        | 5.11    | 7.10 - 27.08  | 8.57     |
| <b>Grip Pressure</b> | 14.68        | 5.45    | 5.34 - 31.14  | 20       |

Overall, the SNR results confirm that most signals were captured with high quality. HR showed the highest signal quality, and grip pressure showed the lowest SNR, probably due to sensor contact quality.

In addition, we also assessed the quality of the recorded video data. To verify the integrity of the video recordings, we computed the frame drop rate by comparing the number of frames extracted using OpenCV (VideoCapture function) with the expected number of frames based on the video’s frame rate and duration. This showed a frame drop rate of 0.00%, confirming that the videos were recorded without loss and are suitable for further behavioral or facial expression analysis.

### 6.3 Inter-Rater Reliability

To check the consistency of the labels, we asked an expert to watch some of the videos (30% of the total data) and label them independently based on KSS criteria. Then we compared participants’ self-reported labels with expert annotations using Cohen’s Kappa measures of inter-rater agreement. For the 9-level KSS scale, the unweighted Kappa was  $\kappa = 0.619$ , indicating substantial agreement. To consider the ordinal nature of the rating and penalize larger disagreements more heavily, we used quadratic-weighted Kappa, which resulted in  $\kappa = 0.967$  (95% CI: [0.951, 0.978]), showing almost perfect agreement. For the binned 3-level, we computed the unweighted Kappa, resulting in a  $\kappa = 0.88$ , also reflecting high consistency.

### 6.4 Validation of Multimodal Data

To assess the quality of the UL-DD dataset, we performed a classification task as a form of validation. For each data type and signal, we began by combining the raw

{20}------------------------------------------------

synchronized data from all subjects and sessions. To reduce computational cost and maintain consistency across all modalities, we set the sample rate frequency to 4 Hz for all data types. Missing values were handled separately for each subject and session using linear interpolation with forward fill. After interpolation, the data were normalized using MinMaxScaler. We performed three-level drowsiness classification, labeling the data as Low ( $KSS < 4$ ), Medium ( $4 \leq KSS \leq 6$ ), and High ( $KSS > 6$ ). Since our focus was on validating the dataset, we did not apply any additional pre-processing steps. To understand the value of each modality, we performed both unimodal and multimodal classification.

We used two commonly applied machine learning models aligned with [69]: The Support Vector Machine (SVM) with an RBF kernel ( $C=0.1$ , `gamma='auto'`, `class_weight='balanced'`) and the Random Forest (RF) classifier (`max_depth=5`, `n_estimators=100`, `min_samples_leaf=10`). To evaluate model performance, we used 5-fold cross-validation ( $k=5$ ) to ensure generalizability. Folds were stratified to maintain the distribution of drowsiness levels across subjects, accounting for class imbalance and inter-subject variability.

Figure 8 presents the classification accuracy results for individual modalities and multimodal combinations of modalities. For the multimodal task, we employed an early fusion approach, where features from different data types are integrated before feeding them into the model. In this way, we are evaluating not only the individual modality contribution, but also how well they work together. Biometric signals served as the baseline modality for this comparison. Interestingly, the performance trend was consistent across both models. For unimodal data, the RF model performed slightly better, specifically for biometric signals, grip pressure, telemetry, and FAUs. On the other hand, SVM showed stronger results when multiple data types were combined. The most effective combination was biometric signals, behavioral data, and facial features, demonstrating the strength of multimodal learning and the dataset’s validity for drowsiness detection.

![Bar chart comparing SVM and RF model accuracy across different data types and combinations.](ebce355620876e10f907f8b71926c112_img.jpg)

| Modality / Combination | SVM (%) | RF (%) |
|------------------------|---------|--------|
| Biometric              | 63%     | 71%    |
| Telemetry              | 40%     | 45%    |
| Grip                   | 48%     | 54%    |
| Pose                   | 68%     | 65%    |
| FAU                    | 52%     | 59%    |
| Bio + Fac              | 73%     | 71%    |
| Bio + Beh              | 80%     | 74%    |
| Bio + Beh + Fac        | 84%     | 75%    |

Bar chart comparing SVM and RF model accuracy across different data types and combinations.

**Fig. 8:** Comparing the performance of SVM and RF models for different data types and combinations.

{21}------------------------------------------------

## 7 Usage notes

The original version of the UL-DD dataset can be accessed upon request. The data is organized by participants and session, stored in CSV and MP4 formats for ease of use. For pre-processing, we recommend the mentioned steps in Section 5.1. The dataset can be used to develop and test multimodal drowsiness detection models, analyze the relationships between facial, behavioral, and biometric indicators of drowsiness.

## 8 Code availability

Readers can access all the code and tutorials along with the dataset. This includes scripts for combining raw data of all subjects, adjusting sample frequency, assembling datasets by combining different modalities, and labeling. Additionally, the scripts for both unimodal and multimodal analysis are available in the repository. We also provided scripts for the biometric technical analysis and feature extraction code for video data, including 2D/3D FLs, FAUs, and PLs. A detailed README file is included for more information and guidance.

## 9 Limitation and future work

The UL-DD dataset offers a comprehensive and versatile multimodal resource for detecting early signs of drowsiness, yet it comes with several limitations. The first constraint that comes to attention is the number of participants with unbalanced gender population that limits the diversity and may not be generalizable for the broader population. Additionally, using a driving simulator in a controlled lab environment is different from real-world conditions that come with different lighting, weather, traffic, and road conditions. The technical constraints include different signal qualities across modalities and the absence of video data for some subjects, which affects the usage of data for some multimodal analyses, even though extracted features were provided to address this gap. There were recruitment challenges due to the inability to offer incentives, which reduced the participant rate and gender balance.

For future work, we will address these limitations by expanding the number of participants and improving diversity. Collecting real-world data beyond 40 minutes to include different environmental conditions, such as different lighting and occlusion, would enhance the validity. Adding more sensors like seat pressure to capture weight distribution or advanced devices for capturing brain and eye activity could enrich the multimodal aspect of the dataset. The publicity of the dataset will provide opportunities for future research to build robust and generalizable drowsiness detection models by leveraging its diverse signals. These advancements could boost UL-DD’s contributions to driver monitoring systems and road safety research.

## 10 Acknowledgments

This work has been partially funded by NSF (CVDI 10a.005.UL\_TAU) and TietoEvry.

{22}------------------------------------------------

## 11 Competing Interests

The authors declare no competing interests.

## References

- [1] Magana, V.C., Munoz-Organero, M.: Toward safer highways: predicting driver stress in varying conditions on habitual routes. *IEEE vehicular technology magazine* **12**(4), 69–76 (2017)
- [2] Weng, C.-H., Lai, Y.-H., Lai, S.-H.: Driver drowsiness detection via a hierarchical temporal deep belief network. In: *Computer Vision–ACCV 2016 Workshops: ACCV 2016 International Workshops, Taipei, Taiwan, November 20–24, 2016, Revised Selected Papers, Part III* 13, pp. 117–133 (2017). Springer
- [3] Salzillo, G., Natale, C., Fioccola, G.B., Landolfi, E.: Evaluation of driver drowsiness based on real-time face analysis. In: *2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)*, pp. 328–335 (2020). IEEE
- [4] Hwang, T., Kim, M., Hong, S., Park, K.S.: Driver drowsiness detection using the in-ear eeg. In: *2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)*, pp. 4646–4649 (2016). IEEE
- [5] Massoz, Q., Langohr, T., François, C., Verly, J.G.: The ulg multimodality drowsiness database (called drozy) and examples of use. In: *2016 IEEE Winter Conference on Applications of Computer Vision (WACV)*, pp. 1–7 (2016). IEEE
- [6] Zheng, W.-L., Lu, B.-L.: A multimodal approach to estimating vigilance using eeg and forehead eog. *Journal of neural engineering* **14**(2), 026017 (2017)
- [7] Shen, M., Zou, B., Li, X., Zheng, Y., Li, L., Zhang, L.: Multi-source signal alignment and efficient multi-dimensional feature classification in the application of eeg-based subject-independent drowsiness detection. *Biomedical Signal Processing and Control* **70**, 103023 (2021)
- [8] Esteves, T., Pinto, J.R., Ferreira, P.M., Costa, P.A., Rodrigues, L.A., Antunes, I., Lopes, G., Gamito, P., Abrantes, A.J., Jorge, P.M., *et al.*: Automotive: a case study on automatic multimodal drowsiness detection for smart vehicles. *IEEE Access* **9**, 153678–153700 (2021)
- [9] Schleicher, R., Galley, N., Briest, S., Galley, L.: Blinks and saccades as indicators of fatigue in sleepiness warnings: looking tired? *Ergonomics* **51**(7), 982–1010 (2008)
- [10] Ghoddoosian, R., Galib, M., Athitsos, V.: A realistic dataset and baseline temporal model for early drowsiness detection. In: *Proceedings of the Ieee/cvf*

{23}------------------------------------------------

- Conference on Computer Vision and Pattern Recognition Workshops, pp. 0–0 (2019)
- [11] Abtahi, S., Omidyeganeh, M., Shirmohammadi, S., Hariri, B.: Yawdd: A yawning detection dataset. In: Proceedings of the 5th ACM Multimedia Systems Conference, pp. 24–28 (2014)
- [12] Chiou, C.-Y., Wang, W.-C., Lu, S.-C., Huang, C.-R., Chung, P.-C., Lai, Y.-Y.: Driver monitoring using sparse representation with part-based temporal face descriptors. *IEEE transactions on intelligent transportation systems* **21**(1), 346–361 (2019)
- [13] Yılmaz, E.K., Akcayol, M.A.: Sust-ddd: A real-drive dataset for driver drowsiness detection. In: Conference of Open Innovations Association (FRUCT), vol. 6 (2022)
- [14] Jegham, I., Khalifa, A.B., Alouani, I., Mahjoub, M.A.: A novel public dataset for multimodal multiview and multispectral driver distraction analysis: 3mdad. *Signal Processing: Image Communication* **88**, 115960 (2020)
- [15] Fusek, R.: Pupil localization using geodesic distance. In: Advances in Visual Computing: 13th International Symposium, ISVC 2018, Las Vegas, NV, USA, November 19–21, 2018, Proceedings 13, pp. 433–444 (2018). Springer
- [16] Nuralif, I., Yuniarno, E.M., Suprapto, Y.K., Wicaksono, A.A.: Driver fatigue detection based on face mesh features using deep learning. In: 2023 International Seminar on Intelligent Technology and Its Applications (ISITIA), pp. 1–5 (2023). IEEE
- [17] Roth, M., Gavrila, D.M.: Dd-pose-a large-scale driver head pose benchmark. In: 2019 IEEE Intelligent Vehicles Symposium (IV), pp. 927–934 (2019). IEEE
- [18] Ohn-Bar, E., Trivedi, M.: The power is in your hands: 3d analysis of hand gestures in naturalistic video. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 912–917 (2013)
- [19] Choi, H.-T., Back, M.-K., Lee, K.-C.: Driver drowsiness detection based on multimodal using fusion of visual-feature and bio-signal. In: 2018 International Conference on Information and Communication Technology Convergence (ICTC), pp. 1249–1251 (2018). IEEE
- [20] Friedrichs, F., Yang, B.: Camera-based drowsiness reference for driver state classification under real driving conditions. In: 2010 IEEE Intelligent Vehicles Symposium, pp. 101–106 (2010). IEEE
- [21] Faraji, F., Lotfi, F., Khorramdel, J., Najafi, A., Ghaffari, A.: Drowsiness detection based on driver temporal behavior using a new developed dataset. *arXiv preprint*

{24}------------------------------------------------

arXiv:2104.00125 (2021)

- [22] Yang, L., Yang, H., Wei, H., Hu, Z., Lv, C.: Video-based driver drowsiness detection with optimised utilization of key facial features. *IEEE Transactions on Intelligent Transportation Systems* (2024)
- [23] Ortega, J.D., Kose, N., Cañas, P., Chao, M.-A., Unnervik, A., Nieto, M., Otaegui, O., Salgado, L.: Dmd: A large-scale multi-modal driver monitoring dataset for attention and alertness analysis. In: *Computer Vision–ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part IV* 16, pp. 387–405 (2020). Springer
- [24] Das, K., Sharak, S., Riani, K., Abouelenien, M., Burzo, M., Papakostas, M.: Multimodal detection of drivers drowsiness and distraction. In: *Proceedings of the 2021 International Conference on Multimodal Interaction*, pp. 416–424 (2021)
- [25] Guo, Z., Zhou, M., Li, G.: Driver vigilance state estimation based on multisource data. *IEEE Sensors Journal* **23**(3), 2592–2604 (2022)
- [26] Shi, J., Wang, K.: Fatigue driving detection method based on time-space-frequency features of multimodal signals. *Biomedical Signal Processing and Control* **84**, 104744 (2023)
- [27] Kalanadhabhatta, M., Min, C., Montanari, A., Kawsar, F.: Fatigueset: A multi-modal dataset for modeling mental fatigue and fatigability. In: *International Conference on Pervasive Computing Technologies for Healthcare*, pp. 204–217 (2021). Springer
- [28] Stappen, L., Baird, A., Schumann, L., Schuller, B.: The multimodal sentiment analysis in car reviews (muse-car) dataset: Collection, insights and improvements. *IEEE Transactions on Affective Computing* **14**(2), 1334–1350 (2021)
- [29] Tao, X., Gao, D., Zhang, W., Liu, T., Du, B., Zhang, S., Qin, Y.: A multimodal physiological dataset for driving behaviour analysis. *Scientific data* **11**(1), 378 (2024)
- [30] Clark, E.A., Kessinger, J., Duncan, S.E., Bell, M.A., Lahne, J., Gallagher, D.L., O’Keefe, S.F.: The facial action coding system for characterization of human affective response to consumer product-based stimuli: a systematic review. *Frontiers in psychology* **11**, 920 (2020)
- [31] Vicente, J., Laguna, P., Bartra, A., Bailón, R.: Drowsiness detection using heart rate variability. *Medical & biological engineering & computing* **54**, 927–937 (2016)
- [32] Raymann, R.J., Swaab, D.F., Van Someren, E.J.: Skin temperature and sleep-onset latency: changes with age and insomnia. *Physiology & behavior* **90**(2-3),

{25}------------------------------------------------

- [33] Dolson, C.M., Harlow, E.R., Phelan, D.M., Gabbett, T.J., Gaal, B., McMellen, C., Geletka, B.J., Calcei, J.G., Voos, J.E., Seshadri, D.R.: Wearable sensor technology to predict core body temperature: A systematic review. *Sensors* **22**(19), 7639 (2022)
- [34] Chowdhury, A., Shankaran, R., Kavakli, M., Haque, M.M.: Sensor applications and physiological features in drivers' drowsiness detection: A review. *IEEE sensors Journal* **18**(8), 3055–3067 (2018)
- [35] Jiao, Y., Zhang, C., Chen, X., Fu, L., Jiang, C., Wen, C.: Driver fatigue detection using measures of heart rate variability and electrodermal activity. *IEEE Transactions on Intelligent Transportation Systems* **25**(6), 5510–5524 (2023)
- [36] Lara, O.D., Labrador, M.A.: A survey on human activity recognition using wearable sensors. *IEEE communications surveys & tutorials* **15**(3), 1192–1209 (2012)
- [37] Saleem, A.A., Siddiqui, H.U.R., Raza, M.A., Rustam, F., Dudley, S., Ashraf, I.: A systematic review of physiological signals based driver drowsiness detection systems. *Cognitive neurodynamics* **17**(5), 1229–1259 (2023)
- [38] Forcolin, F., Buendia, R., Candefjord, S., Karlsson, J., Sjöqvist, B.A., Anund, A.: Comparison of outlier heartbeat identification and spectral transformation strategies for deriving heart rate variability indices for drivers at different stages of sleepiness. *Traffic injury prevention* **19**(sup1), 112–119 (2018)
- [39] Clarke, G.W., Chan, A.D., Adler, A.: Effects of motion artifact on the blood oxygen saturation estimate in pulse oximetry. In: 2014 IEEE International Symposium on Medical Measurements and Applications (MeMeA), pp. 1–4 (2014). IEEE
- [40] Thomson, S., Morrell, M.J., Cordingley, J.J., Semple, S.J.: Ventilation is unstable during drowsiness before sleep onset. *Journal of applied physiology* **99**(5), 2036–2044 (2005)
- [41] Dong, Y., Hu, Z., Uchimura, K., Murayama, N.: Driver inattention monitoring system for intelligent vehicles: A review. *IEEE transactions on intelligent transportation systems* **12**(2), 596–614 (2010)
- [42] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M.G., Lee, J., et al.: Mediapipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172* (2019)
- [43] Mahomed, A.S., Saha, A.K.: Driver posture recognition: A review. *IEEE Access* (2024)

{26}------------------------------------------------

- [44] LEE, K., HYUN, S.-A., Oah, S.: Detecting driver fatigue by steering wheel grip force. *International Journal of Contents* **12**(1), 44–48 (2016)
- [45] Baulk, S.D., Reyner, L., Horne, J.A.: Driver sleepiness—evaluation of reaction time measurement as a secondary task. *Sleep* **24**(6), 695–698 (2001)
- [46] Åkerstedt, T., Gillberg, M.: Subjective and objective sleepiness in the active individual. *International journal of neuroscience* **52**(1-2), 29–37 (1990)
- [47] Tavakol, M., Dennick, R.: Making sense of cronbach’s alpha. *International journal of medical education* **2**, 53 (2011)
- [48] King, D.E.: Dlib-ml: A machine learning toolkit. *The Journal of Machine Learning Research* **10**, 1755–1758 (2009)
- [49] Iwama, S., Takemi, M., Eguchi, R., Hirose, R., Morishige, M., Ushiba, J.: Two common issues in synchronized multimodal recordings with eeg: Jitter and latency. *Neuroscience Research* **203**, 1–7 (2024)
- [50] Dolmans, T., Poel, M., Klooster, J., Veldkamp, B.: Data synchronisation and processing in multimodal research. *Measuring Behavior 2020-21 Volume 1*, 26–32 (2020)
- [51] Kreibig, S.D.: Autonomic nervous system activity in emotion: A review. *Biological psychology* **84**(3), 394–421 (2010)
- [52] Zhang, N., Fard, M., Bhuiyan, M., Verhagen, D., Azari, M., Robinson, S.: The effects of physical vibration on heart rate variability as a measure of drowsiness. *Ergonomics* **61**(9), 1259–1272 (2018)
- [53] Bodaghi, M., Hosseini, M., Gottumukkala, R.: A multimodal intermediate fusion network with manifold learning for stress detection. In: 2024 IEEE 3rd International Conference on Computing and Machine Intelligence (ICMI), pp. 1–8 (2024). IEEE
- [54] Hosseini, M., Bodaghi, M., Bhupatiraju, R.T., Maida, A., Gottumukkala, R.: Multimodal stress detection using facial landmarks and biometric signals. *arXiv preprint arXiv:2311.03606* (2023)
- [55] Bodaghi, M.: Adaptive attention-based lstm framework for multimodal driver drowsiness prediction. Master’s thesis, University of Louisiana at Lafayette (2024)
- [56] Cao, Z., Hidalgo, G., Simon, T., Wei, S.-E., Sheikh, Y.: Openpose: Realtime multi-person 2d pose estimation using part affinity fields. *IEEE Transactions on Pattern Analysis & Machine Intelligence* **43**(01), 172–186 (2021)
- [57] Bajaj, J.S., Kumar, N., Kaushal, R.K., Gururaj, H., Flammini, F., Natarajan,

{27}------------------------------------------------

- R.: System and method for driver drowsiness detection using behavioral and sensor-based physiological measures. *Sensors* **23**(3), 1292 (2023)
- [58] Pourbemany, J., Zhu, Y., Bettati, R.: A survey of wearable devices pairing based on biometric signals. *IEEE Access* **11**, 26070–26085 (2023)
- [59] Zhu, L.: Emotion detection system using electrodermal activity signals from wearable devices with deep learning techniques. PhD thesis, University of Guelph (2024)
- [60] Zhang, F., Shao, Y., Sun, Y., Gao, C., Sang, N.: Self-supervised low-light image enhancement via histogram equalization prior. In: *Chinese Conference on Pattern Recognition and Computer Vision (PRCV)*, pp. 63–75 (2023). Springer
- [61] Yang, J., Xu, Y., Yue, H., Jiang, Z., Li, K.: Low-light image enhancement based on retinex decomposition and adaptive gamma correction. *IET image processing* **15**(5), 1189–1202 (2021)
- [62] Li, S., Bi, X., Zhao, Y., Bi, H.: Extended neighborhood-based road and median filter for impulse noise removal from depth map. *Image and vision computing* **135**, 104709 (2023)
- [63] Royston, P.: Approximating the shapiro-wilk w-test for non-normality. *Statistics and computing* **2**, 117–119 (1992)
- [64] McKight, P.E., Najab, J.: Kruskal-wallis test. *The corsini encyclopedia of psychology*, 1–1 (2010)
- [65] Pinheiro, J., Bates, D.: *Mixed-effects Models in S and S-PLUS*. Springer, ??? (2006)
- [66] Kräuchi, K., Wirz-Justice, A.: Circadian clues to sleep onset mechanisms. *Neuropsychopharmacology* **25**(1), 92–96 (2001)
- [67] Tai, Y., Obayashi, K., Yamagami, Y., Saeki, K.: Association between circadian skin temperature rhythms and actigraphic sleep measures in real-life settings. *Journal of Clinical Sleep Medicine* **19**(7), 1281–1292 (2023)
- [68] Szymusiak, R.: Body temperature and sleep. *Handbook of clinical neurology* **156**, 341–351 (2018)
- [69] Yang, P., Liu, N., Liu, X., Shu, Y., Ji, W., Ren, Z., Sheng, J., Yu, M., Yi, R., Zhang, D., *et al.*: A multimodal dataset for mixed emotion recognition. *Scientific Data* **11**(1), 847 (2024)
- [70] Patterson, J.A., McIlwraith, D.C., Yang, G.-Z.: A flexible, low noise reflective ppg sensor platform for ear-worn heart rate monitoring. In: *2009 Sixth International Workshop on Wearable and Implantable Body Sensor Networks*, pp. 286–291

{28}------------------------------------------------

(2009). IEEE

- [71] Posada-Quintero, H.F., Chon, K.H.: Innovations in electrodermal activity data collection and signal processing: A systematic review. *Sensors* **20**(2), 479 (2020)
- [72] Day, E.M., Alcantara, R.S., McGeehan, M.A., Grabowski, A.M., Hahn, M.E.: Low-pass filter cutoff frequency affects sacral-mounted inertial measurement unit estimations of peak vertical ground reaction force and contact time during treadmill running. *Journal of Biomechanics* **119**, 110323 (2021)