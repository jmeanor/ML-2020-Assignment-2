Assignment 2
Randomized Optimization
Spring 2020
John Meanor

To run the project:
1. Using Python 3.7+, simply install all dependencies/imports.
2. Run `python3 main.py`

The code will generate an output directory with subdirectories for each execution of the program.
Each subdirectory will contain subdirectories for each section of the analysis. A metadata.txt log file will contain the metrics collected to perform the analysis in the paper. Additionally, graphs will be generated in a PNG image format for fitness curve evaluation.

Note: some code blocks have been commented out that were previously used during experimentation for the analysis. Simply uncomment the code blocks desired to obtain output for those trials.

Project File structure 
.
├── Part1.py
├── Part2.py
├── README.txt
├── graph.py
├── input
│   └── hirosaki_temp_cherry.csv
├── main.py
├── myLogger.py
└── tutorial.py