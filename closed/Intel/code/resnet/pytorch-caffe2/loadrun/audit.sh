create_audit_folder(){
for tests in TEST01 TEST03 TEST04-A TEST04-B TEST05
do
for model in resnet mobilenet
do
mkdir -p audit/$model/Server/$tests/accuracy
mkdir -p audit/$model/Server/$tests/performance
mkdir -p audit/$model/Offline/$tests/accuracy
mkdir -p audit/$model/Offline/$tests/performance
mkdir -p audit/$model/SingleStream/$tests/accuracy
mkdir -p audit/$model/SingleStream/$tests/performance

done
done
}

create_audit_folder
. submit.sh
. test01.sh
. test03.sh
. test04.sh
. test05.sh
