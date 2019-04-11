# Road traffic monitoring

[Module 6][m6] Project from [Master in Computer Vision Barcelona][master] program.

The goal of this project is to learn the basic concepts and techniques related to video
sequences mainly for surveillance applications. More particularly, we develop techniques to track vehicles across different cameras which could be used for traffic monitoring (estimate speed, traffic density, etc.).

# Scope
- Use of statistical models to estimate the background information of 
the video sequence
- Use of deep learning techniques to detect the foreground
- Use optical flow estimations and compensations
- Track detections
- Analyze system performance evaluation

# Applicability

Any problem where video sequence analysis can be applied to obtain 
accurate automatic results 

# How to run

Package dependencies are managed using _pipenv_. Check 
[its documentation][pipenv_docs] or [its repository][pipenv_repo].

For every week there are a set of tasks to do. They can be found organized
in package and modules for weeks and tasks respectively, i.e. `week1\task1.py`.
All of them can be run from `main.py` which is used to collect all work done.

# Main tasks per week
- Week1: analysis of the AICity challenge 2019 dataset. Creation of Optical Flow metrics (Mean Squared Error in Non-occluded pixels (MSEN); Percentage of Erroneous Pixels which are Not occluded (PEPN)).
- Week2: background subtraction/foreground detection with single gaussian models: non-adaptive vs adaptive background model.
- Week3: Object Detection (off-the-shelf + fine-tuning). Introduction to Object Tracking (Multi-track *single* camera).
- Week4: Optical Flow estimation (off-the-shelf + block matching-based one). Video stabilization via Optical Flow vs State-of-the-art.
- Week5: improve tracking with Optical Flow. Start thinking how to combine tracks (merge/split/swap IDs) among different cameras (Multi-track *Multiple* Camera).
- Week6: test different multicamera setups and assess MTSC and MTMC against the state-of-the-art presented in the challenge's paper.

The final presentation summarizing the techniques employed for vehicle tracking and re-identification can be [accessed here][slides].

[slides]: https://docs.google.com/presentation/d/1aa3_eHkDxvxJ-yO88M23ssG38caLPWcExaC5K7iYHiU/edit?usp=sharing
[m6]: http://pagines.uab.cat/mcv/content/m6-video-analysis
[master]: http://pagines.uab.cat/mcv/
[pipenv_docs]: https://pipenv.readthedocs.io/en/latest/install/
[pipenv_repo]: https://github.com/pypa/pipenv
