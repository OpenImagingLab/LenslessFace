python tools/analysis_tools/analyze_logs.py plot_curve $1 --keys "accuracy" 

python tools/analysis_tools/analyze_logs.py plot_curve $1  

python tools/analysis_tools/gen_psf_gif.py $(dirname "$1")