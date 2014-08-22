data_path=$1
ranks=$2
hostsfile='mpd.hosts'
rows=$3
threads=$4
chunkSize=$5
#rows=($(wc -l $data_path | cut -f1 -d' '))
echo "rows="$rows
rowsPerSlice=$((1+$rows/($ranks-1)))
echo "rowsPerSlice="$rowsPerSlice
currentDir=`pwd`
echo 'Preparing the data...' 
rm -r prepared_data
mkdir prepared_data
cp $data_path  prepared_data
#echo "split -a 4 -d -l $rowsPerSlice $data_path prepared_data/chunk"
#split -a 4 -d -l $rowsPerSlice $data_path prepared_data/chunk
#echo 'Done preparing the data.' 
echo 'creating class vectors...'
awk -F "," '
{
	i=NF;
	printf "%s\n",$i >> "prepared_data/classvector1";
	#printf "%s\n",$i >> "prepared_data/classvector2";
}' $data_path

echo 'Copying executables to slave nodes. Note that the path ' $currentDir ' should exist in all slave nodes'
i=0
dataFileName=`basename $data_path`
cat $hostsfile | while read line
do
if [ $i -gt 0 ]
then
	echo $line
	
	#fileId=`printf '%04i\n' $[i-1]`
	#ssh $line "rm -r ${currentDir}; mkdir ${currentDir}; mkdir ${currentDir}/prepared_data" < /dev/null
	scp $currentDir'/fuzz' $line:$currentDir'/fuzz' 
	#scp $currentDir'/prepared_data/chunk'$fileId $line:$currentDir'/prepared_data/'$dataFileName
	scp $currentDir'/'$dataFileName $line:$currentDir'/'$dataFileName
fi
	i=$[i+1]
done
echo 'Done copying.'
columns=$(awk -F "," 'NR==1{ print NF-1}'  "$data_path")
echo 'columns='$columns
rm -r result
mkdir result
numOfClasses=$(ls prepared_data/ | grep classvector | wc -l)
echo "num of classes="$numOfClasses
echo mpiexec.hydra -rr -f $currentDir/$hostsfile -n $ranks $currentDir/fuzz  $currentDir/$dataFileName $currentDir/prepared_data/ $columns $rows $threads $numOfClasses $threads
mpiexec.hydra -rr -f $currentDir/$hostsfile -n $ranks $currentDir/fuzz  $currentDir/$dataFileName $currentDir/prepared_data/ $columns $rows $threads $numOfClasses $threads
