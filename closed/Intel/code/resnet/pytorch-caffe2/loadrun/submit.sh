sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val\/\" \\/g' netrun.sh
sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val\/\" \\/g' loadrun.sh
sed -i 's/--images \".*\" \\/--images \"\/home\/mlt\/tools\/val\/\" \\/g' single_stream.sh
sed -i 's/itime=.*/itime=1/g' run.sh
sed -i "s/accuracy=.*/accuracy=1/" run.sh
sed -i "s/performance=.*/performance=0/" run.sh

if [ -f "audit.config" ];then
 rm audit.config
fi
rm -rf results
. run.sh
mv results submit

