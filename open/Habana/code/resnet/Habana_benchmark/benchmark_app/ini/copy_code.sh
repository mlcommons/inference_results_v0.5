code_base_dir="$1"
base_dir='./habana_submission0.5/closed/Habana/'
benchmarks='ssd-large resnet'
measurements='/measurements/Goya_1'
scenarios='Offline SingleStream Server MultiStream'
for bm in $benchmarks; do
    mkdir --parents $base_dir'/code/'$bm
    cp $code_base_dir  $base_dir'/code/'$bm  -r
    #rm $base_dir'/code/'$bm'/Habana_benchmark/runner/source' -r
    #rm $base_dir'/code/'$bm'/Habana_benchmark/runner/CMakeLists.txt' 
    rm --force README.md
    echo '1. For building the benchmark code please consult the README.dm file under '$base_dir'/code/'$bm'/Habana_benchmark/' > README.md
    echo '   This files gives explantion about usage of the preprocessing functions for ssd-large and resnet50 - building model files' >> README.md27
    echo '   and image preprocess needed to be performed before running the benchmark.' >> README.md
    echo '2. User may consult the Habana_mlperf_benchmark_guide.pdf placed under'$base_dir'/code/'$bm'/Habana_benchmark/' >> README.md
    echo '   The Habana_mlperf_benchmark_guide.pdf give a details overview about the connection between loadgen and Habna bechnmark code.' >> README.md

    cp   README.md $base_dir'/code/'$bm
done
for bm in $benchmarks; do
    for sc in $scenarios; do
        rm --force README.md 
        echo 'For building the benchmark code please consulte the README.dm file under '$base_dir'/code/'$bm > README.md
        echo 'To run the benchmark code on '$bm' model in '$sc' scenario please use the following command line' >> README.md
        echo './HABANA_benchmark_app '  $bm'_'$sc'_submissionrun.ini' >> README.md
        cp README.md $base_dir/$measurements'/'$bm'/'$sc'/'
    done
done
