/*
 *
 *  Created on: Jul 12, 2014
 *      Author: Hasan Asfoor
 *      *****************************************************************************
 *      In this version, the same dataset file is assumed to be local to both master and slave. This is the final version of the code.
 *      This is the final version.
 */

#include<pthread.h>
#include<stdlib.h>
#include <math.h>
#include<float.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <err.h>
#include <ctype.h>
#   define errno (*__errno_location ())
# define STRING_ATTR_SIZE 100

/*
 * A structure representing a set of parsed rows
 */
typedef struct {
	int numOfStringAttr;
	int numOfNumericAttr;
	double* attrN;
	char* attrS;
	long numOfRows;
} Rows;

/*
 * Arguments used to be passed to a function when computing min and max for ranges
 */
typedef struct {
	long numOfColumns;
	double* max;
	double* min;
} RangesArguments;

/*
 * Argument used to be passed to the loop parallel function
 */
typedef struct {
	long start, end, increment;
	void* args;
	int (*function)(long, long, long, int, void*);
	int threadId;

} LoopArguments;

/*
 * Argument used to be passed to the row processing function
 */
typedef struct {
	Rows currentRow;
	double* ranges;
	Rows rows;
	int numOfColumns;
	long numOfRows;
	int numOfClasses;
	long chunkSize;
	long startIndex;
	long endIndex;
	double* uApprox;
	double* lApprox;
	double* cvector;
	int numOfThreads;
	long rowsCompleted;
	long startLine;
	char* lines;
	int* indices;
	int rank;
	time_t* start_time;//, now, end_time;
} ProcessRowArgs;

int isNumeric (const char * s)
{
    if (s == NULL || *s == '\0' || isspace(*s))
      return 0;
    char * p;
    strtod (s, &p);
    return *p == '\0';
}


/*
 * INitializes a set of rows
 */
static int initRows(Rows* rows, long numOfRows, int numOfStrAttr, int numOfNumericAttr)
{
	rows->numOfRows = numOfRows;
	rows->numOfNumericAttr = numOfNumericAttr;
	rows->attrN = malloc(sizeof(double)*numOfNumericAttr*numOfRows);
	memset(rows->attrN,0,sizeof(double)*numOfNumericAttr*numOfRows);
	rows->numOfStringAttr = numOfStrAttr;
	rows->attrS = malloc(sizeof(char*)*numOfStrAttr*STRING_ATTR_SIZE*numOfRows);
	memset(rows->attrS,0,sizeof(char*)*numOfStrAttr*STRING_ATTR_SIZE*numOfRows);
	return 1;
}

/*
 * frees the content of a Rows structure
 */
static int freeRowsContent(Rows* rows)
{
	free(rows->attrN);
	free(rows->attrS);
	return 1;
}

/*
 * add a numberic attribute value to a Rows structure
 */
static int addNumericAttr(Rows* rows, long rowNum, double attrVal, int attrIndex)
{
	rows->attrN[rowNum * rows->numOfNumericAttr + attrIndex] = attrVal;
	return 1;
}

/*
 * add a string attribute value to a Rows structure
 */
static int addStringAttr(Rows* rows, long rowNum, char* attrVal, int attrIndex)
{
	char* ptr = rows->attrS + rowNum * rows->numOfStringAttr * STRING_ATTR_SIZE + attrIndex * STRING_ATTR_SIZE;
	sprintf(ptr, "%s",attrVal);
	return 1;
}

static char* getStrAttrOfRow(Rows* rows, long rowNum, int attrNum)
{
	char* ptr = rows->attrS;
	ptr+=rowNum * rows->numOfStringAttr * STRING_ATTR_SIZE + attrNum * STRING_ATTR_SIZE;
	return ptr;
}

//attrNum represents the numeric attribute number
static double getNumAttrOfRow(Rows* rows,long rowNum, int attrNum)
{
	double result = -1;
	result = rows->attrN[rowNum * rows->numOfNumericAttr + attrNum];
	return result;
}

/*
 * parse a string line and add it to a Rows structure
 */
static int addLineToRows(Rows* rows, int rowNum, char* line, double* ranges)
{
	int numAttrs = rows->numOfNumericAttr;
	int strAttrs = rows->numOfStringAttr;
	int numOfColumns = numAttrs+ strAttrs;
	char *token, *saveptr;
	double numVal;
	long i = 0,j=0;
	char buff[strlen(line) + 1];
	strcpy(buff, line);
	token = strtok_r(buff, ",", &saveptr);
	while (token != NULL && i+j < numOfColumns) {
		if(isNumeric(token))
		{
			sscanf(token, "%lf", &numVal);
			numVal = numVal*ranges[i];
			addNumericAttr(rows,rowNum,numVal,i);
			i++;
		}
		else
		{
			addStringAttr(rows,rowNum,token,j);
			j++;
		}
		token = strtok_r(NULL, ",", &saveptr);
	}
	//printf("done tokenizing\n");*/
	return 1;
}

static int getRowAt(Rows* rows, int rowNum, Rows* result)
{
	result->numOfRows = 1;
	result->numOfNumericAttr = rows->numOfNumericAttr;
	result->numOfStringAttr = rows->numOfStringAttr;
	//optimize 5: need to find a way to reduce the multiplications here
	result->attrN = rows->attrN + rowNum * rows->numOfNumericAttr;
	result->attrS = rows->attrS + rowNum * rows->numOfStringAttr * STRING_ATTR_SIZE;
	return 1;
}

static int printRow(Rows* rows, int rowNum)
{
	int i=0;
	for (; i < rows->numOfNumericAttr; i++) {
		printf("%.2lf",rows->attrN[rowNum * rows->numOfNumericAttr + i]);
		if(i < rows->numOfNumericAttr -1)
			printf(",");
	}
	if(rows->numOfStringAttr>0)
		printf(",");
	char* strptr = rows->attrS + rowNum * rows->numOfStringAttr * STRING_ATTR_SIZE;
	for (i = 0; i < rows->numOfStringAttr; i++) {
		strptr +=i*STRING_ATTR_SIZE;
		printf("%s", strptr);
		if (i < rows->numOfStringAttr - 1)
			printf(",");

	}

	printf("\n");
	return 1;
}

static int printAllRows(Rows* rows)
{
	long i=0;
	for(;i<rows->numOfRows;i++)
		printRow(rows,i);
	return 1;
}


/*
 * executes a thread
 */
static int runThreads(void* args) {
//printf("runTHread\n");
	LoopArguments *arg = (LoopArguments*) args;
	long start = arg->start;
	long end = arg->end;
	long increment = arg->increment;
	void* userArg = arg->args;
	int (*function)(long, long, long, int, void*) = arg->function;
	int tid = arg->threadId;
	function(start, end, increment, tid, userArg);
	pthread_exit(0);
	return 0;
}

/*
 * Runs several iterations of a loop in parallel by multiple threads
 */
static int parallelLoop(long start, long end, long increment, long numOfThreads, int (*function)(long, long, long, int, void*), void* args) {
	pthread_t thread[numOfThreads];
	pthread_attr_t attr;
	int rc;
	long t;
	void *status;
	LoopArguments loopArgs[numOfThreads];

	/* Initialize and set thread detached attribute */
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	int iterPerThread = (end - start) / numOfThreads;

	iterPerThread -= (iterPerThread % increment);

	for (t = 0; t < numOfThreads; t++) {
		loopArgs[t].increment = increment;
		loopArgs[t].args = args;
		loopArgs[t].function = function;
		loopArgs[t].threadId = t;

		loopArgs[t].start = start + t * iterPerThread;
		if (t == numOfThreads - 1)
			loopArgs[t].end = end;
		else {
			loopArgs[t].end = loopArgs[t].start + iterPerThread;
			if (loopArgs[t].end > end)
				loopArgs[t].end = end;
		}

		rc = pthread_create(&thread[t], &attr, runThreads, (&loopArgs[t]));
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
	}

	/* Free attribute and wait for the other threads */
	pthread_attr_destroy(&attr);
	for (t = 0; t < numOfThreads; t++) {
		rc = pthread_join(thread[t], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}

	}
	return 1;

}

static int initArray(double* arr, double value, long size) {
	long ind = 0;
	for (ind = 0; ind < size; ind++) {
		arr[ind] = value;
	}
	return 1;
}
static int printArray(double* arr, long size, char* arrName, int rank) {
	long ind = 0;
	for (ind = 0; ind < size; ind++) {
		printf("rank%d: %s[%d]=%lf\n", rank, arrName, ind, arr[ind]);
	}
	return 1;
}
static long determine_file_size(char *filename) {
	FILE *fd;
	long size = 0;

	fd = fopen(filename, "r");
	if (fd == NULL) {
		printf("%s: Could not open %s to read - (%s)\n", __func__, filename, strerror(errno));
		return (long) errno;
	}
	fseek(fd, 0L, SEEK_END);
	size = ftell(fd);
	rewind(fd);
	fclose(fd);
	return size;
}

/*
 * Read all lines of a file and pass each line to a function
 */
static int readLines(int fd, const char * fname, int (*call_back)(int, long, const char*, void*), void* param, int rank) {
	struct stat fs;
	char *buf, *buf_end;
	char *begin, *end, c;
	char line[1000000];
//printf("param=%d\n",param);
	long lineNumber = 0;
	if (fd == -1) {
		err(1, "open: %s", fname);
		return 0;
	}

	if (fstat(fd, &fs) == -1) {
		err(1, "stat: %s", fname);
		return 0;
	}

	/* fs.st_size could have been 0 actually */
	buf = mmap(0, fs.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (buf == (void*) -1) {
		err(1, "mmap: %s", fname);
		close(fd);
		return 0;
	}

	buf_end = buf + fs.st_size;

	begin = end = buf;
	while (1) {
		if (!(*end == '\r' || *end == '\n')) {
			if (++end < buf_end)
				continue;
		} else if (1 + end < buf_end) {
			/* see if we got "\r\n" or "\n\r" here */
			c = *(1 + end);
			if ((c == '\r' || c == '\n') && c != *end)
				++end;
		}

		/* call the call back and check error indication. Announce
		 error here, because we didn't tell call_back the file name */

//memcpy(begin, line, end - begin);
//int ind = end - begin;
		strncpy(line, begin, end - begin);
		line[end - begin] = '\0';

//write(fileno(stdout), begin, end - begin + 1);
		if (!call_back(rank, lineNumber, line, param)) {
			err(1, "[callback] %s", fname);
			break;
		}

		if ((begin = ++end) >= buf_end)
			break;
		lineNumber++;
	}

	munmap(buf, fs.st_size);
//close(fd);
	return 1;
}

/*
 * Read multiple lines starting at an index and pass each of the read lines to a function
 */
static int readMLinesAt(long startIndex, long numLines,  const char * fname, int (*call_back)(int, long, const char*, void*, int, int*), void* param, int rank) {
	struct stat fs;
	char *buf, *buf_end;
	char *begin, *end, c;
	char lines[1000000];
	int indices[numLines];
//printf("param=%d\n",param);
	long lineNumber = 0;
	int fd = open(fname, O_RDONLY);
	if (fd == -1) {
		err(1, "open: %s", fname);
		return 0;
	}

	if (fstat(fd, &fs) == -1) {
		err(1, "stat: %s", fname);
		return 0;
	}

	/* fs.st_size could have been 0 actually */
	buf = mmap(0, fs.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (buf == (void*) -1) {
		err(1, "mmap: %s", fname);
		close(fd);
		return 0;
	}

	buf_end = buf + fs.st_size;

	begin = end = buf;

	int i=0;

	long count = 0;
	long indexCount=1;
	indices[count] = end-begin;
	char* startChar;
	char* endChar;
	while (1) {
		if (!(*end == '\r' || *end == '\n')) {
			if (++end < buf_end)
				continue;
		} else if (1 + end < buf_end) {
			/* see if we got "\r\n" or "\n\r" here */
			c = *(1 + end);
			if ((c == '\r' || c == '\n') && c != *end)
				++end;
		}



		if(count == startIndex)
		{
			startChar = begin;
		}

		if(count == startIndex+numLines-1)
		{
			endChar = end;
			//printf("start=%d; end=%d\n",startChar, endChar);
			strncpy(lines, startChar, endChar-startChar);
			lines[endChar-startChar] = '\0';

			for(i=0;lines[i]!='\0';i++)
			{
				if(lines[i] == '\n')
				{
					printf("char=%c; i=%d\n", lines[i], i);
					//lines[i] = '\0';
					indices[indexCount] = i+1;
					indexCount++;
				}
			}

			if (!call_back(rank, lineNumber, lines, param, count, indices)) {
				err(1, "[callback] %s", fname);
				break;
			}
			break;
		}

		if (begin >= buf_end || end >= buf_end) {
			//printf("hi");
			break;
		}

		count++;
		begin=++end;

	}



	munmap(buf, fs.st_size);
	close(fd);
	return 1;
}

/*
 * Read a number of lines equal to "numLines" and starts at "startIndex" and stores it on "lines"
 * startIndex: the row index to start at
 * numLines: the number of lines to be read
 * lines: the buffer on which lines are to be stored
 * fname: the filename from which the function reads
 * rank: the rank of the slave node calling the function
 */
static int readChunkFromFile(long startIndex, long numLines,  char* lines, const char * fname, int rank) {
	struct stat fs;
	char *buf, *buf_end;
	char *begin, *end, c;

	int fd = open(fname, O_RDONLY);
	if (fd == -1) {
		err(1, "open: %s", fname);
		return 0;
	}

	if (fstat(fd, &fs) == -1) {
		err(1, "stat: %s", fname);
		return 0;
	}

	/* fs.st_size could have been 0 actually */
	buf = mmap(0, fs.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (buf == (void*) -1) {
		err(1, "mmap: %s", fname);
		close(fd);
		return 0;
	}
	buf_end = buf + fs.st_size;
	begin = end = buf;

	long count = 0;

	char* startChar;
	char* endChar;
	while (1) {
		if (!(*end == '\r' || *end == '\n')) {
			if (++end < buf_end)
				continue;
		} else if (1 + end < buf_end) {
			/* see if we got "\r\n" or "\n\r" here */
			c = *(1 + end);
			if ((c == '\r' || c == '\n') && c != *end)
				++end;
		}



		if(count == startIndex)
		{
			startChar = begin;
		}

		if(count == startIndex+numLines-1)
		{
			endChar = end;
			//printf("start=%d; end=%d\n",startChar, endChar);
			strncpy(lines, startChar, endChar-startChar);
			lines[endChar-startChar] = '\0';
			//printf(lines);
			break;
		}

		if (begin >= buf_end || end >= buf_end) {
			break;
		}

		count++;
		begin=++end;

	}
	munmap(buf, fs.st_size);
	close(fd);
	return 1;
}

/*
 * parse a line and store it on "row"
 */
static int parseLine(char* line, double* row, int numOfColumns) {
	char *token, *saveptr;
	double temp;
	long i = 0;
	char buff[strlen(line) + 1];
	strcpy(buff, line);
	token = strtok_r(buff, ",", &saveptr);
	while (token != NULL && i < numOfColumns) {
		sscanf(token, "%lf", &temp);
		row[i] = temp;
		i++;
		token = strtok_r(NULL, ",", &saveptr);
	}
//printf("done tokenizing\n");
	return 1;
}

static double min(double x, double y) {
	if (x < y)
		return x;
	return y;
}

static double fuzzyAnd(double x, double y) {
	return min(x, y);
}

static double max(double x, double y) {
	if (x > y)
		return x;
	return y;
}

static double fuzzyImply(double x, double y) {
	return max(1 - x, y);
}

static double distanceS(char* str1, char* str2) {
	int result = strcmp(str1, str2);
	if(result == 0)
		return 1;
	else
		return 0;
}

/*
 * Compute the similarity value of row1 and row2
 */
static double getSimilarityOfRows(Rows* row1, Rows* row2) {
	int i = 0;
	double result = 0;
	int numOfColumns = row1->numOfNumericAttr + row1->numOfStringAttr;

	for (i=0; i < row1->numOfNumericAttr; i++) {
		result += fabs(row1->attrN[i]-row2->attrN[i]) ;//* ranges[i];
	}
	result = row1->numOfNumericAttr - result;
	for (i=0; i < row1->numOfStringAttr; i++) {
			result += distanceS(getStrAttrOfRow(row1,0,i), getStrAttrOfRow(row2,0,i));

		}

	return result / numOfColumns;
}

/*
 * reads the content of a file and store it on buffer
 */
static int readEntireFile(char *filename, long size, char **buffer, int rank) {
	size_t nr_read = 0;
	FILE *fh;

	printf("rank%d: %s: filename to open: %s\n", rank, __func__, filename);

	fh = fopen(filename, "r");
	if (fh == NULL) {
		printf("rank%d: %s: Could not open %s to read - (%s)\n", rank, __func__, filename, strerror(errno));
		return -1;
	}

	nr_read = fread(*buffer, size, 1, fh);
	if (nr_read < 1)
		printf("%s: Read %ld out of 1 elements for %s (%s)\n", __func__, nr_read, filename, strerror(errno));
	fclose(fh);

	return 0;
}

/*
 * This function loads all the class vectors in the "path" parameter and store them on "cvector"
 * path: this is the path to the class vector files. It is not the location of a file. Rather, it is a path to a set of file with names classvector1, classvector2...
 * numOfRows: this is the total number of rows in the dataset.
 * numOfClasses: this is the number of class vectors
 * cvector: this is a vector to contain the classvectors. It should be initialized before being passed to this function.
 */
static int loadClassvector(char* path, double* cvector, long numOfRows, int numOfClasses) {
	char tempName[400];
	memset(tempName, 0, 400);
	char* buffer;
	int i = 0,j=0;
	long size = 0;
	long index=-1;
	char *token, *saveptr;
	double temp;
	//Determine the maximum vector size needed to be initialized.
	for (i = 0; i < numOfClasses; i++) {
		sprintf(tempName, "%s%s%d", path, "classvector", i + 1);
		size = max(determine_file_size(tempName), size);
	}
	//allocate memory for the buffer
	buffer = malloc(size);
	//read each class file, parse it and store it on "cvector"
	for (i = 0; i < numOfClasses; i++) {
		sprintf(tempName, "%s%s%d", path, "classvector", i + 1);
		size = determine_file_size(tempName);
		readEntireFile(tempName, size, &buffer, 0);
		token = strtok_r(buffer, "\n", &saveptr);
		j=0;
		while (token != NULL && j < numOfRows) {
			sscanf(token, "%lf", &temp);
			index = i*numOfRows+j;
			cvector[i*numOfRows+j] = (double) temp;
			j++;
			token = strtok_r(NULL, "\n", &saveptr);
		}
	}
	return 1;
}

/*
 * This function computes mins and maxs for one line and stores the result in param->max and param->mins. This is a prep step for computing the ranges
 */
static int computeRanges(int rank, long ln, char* line, void* param) {
	RangesArguments* args = (RangesArguments*) param;
	double *maxs = args->max; //initially store maximums here
	double *mins = args->min;
	int numOfColumns = args->numOfColumns;
	double row[numOfColumns];
	memset(row, 0, numOfColumns);
	parseLine(line, row, numOfColumns);
	int i = 0;
	for (; i < numOfColumns; i++) {

		if (ln == 0) {
			mins[i] = row[i];
			maxs[i] = row[i];
		} else {
			mins[i] = min(mins[i], row[i]);
			maxs[i] = max(maxs[i], row[i]);
		}
	}

	return 1;
}

/*
 * This function runs the master program which basically loads the class vectors and broadcast them. It also broadcast the ranges to slaves as well.
 * dataPath: this is the location of the input dataset file.
 * cvectorPath: this is the path to the class vector files. It is not the location of a file. Rather, it is a path to a set of file with names classvector1, classvector2...
 * numOfRows: this is the total number of rows in the dataset.
 * numOfColumns: this is the number of condition attributes in the dataset.
 * numOfSlaves: this is the number of slave nodes in the cluster
 * numOfClasses: this is the number of class vectors
 * cvector: this is a vector to contain the classvectors. It should be initialized before being passed to this function.
 */
void masters(char* dataPath, char* cvectorPath, long numOfRows, int numOfColumns, int numOfSlaves, int numOfClasses, double* cvector) {

	double mins[numOfColumns];
	double maxs[numOfColumns];
	double ranges[numOfColumns];
	//Load class vectors
	loadClassvector(cvectorPath, cvector, numOfRows, numOfClasses);
	//broadcast the classvectors
	MPI_Bcast(cvector, numOfRows * numOfClasses, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	printf("rank:%d: Class vectors successfully broadcasted..\n", 0);
	//initialize the ranges vector
	initArray(ranges,DBL_MAX,numOfColumns);
	//reduce the mins computed at the slave nodes into the mins vector
	MPI_Reduce(ranges, mins, numOfColumns, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	initArray(ranges,-1 * DBL_MAX,numOfColumns);
	//reduce the max values computed at the slave nodes into the maxs vector
	MPI_Reduce(ranges, maxs, numOfColumns, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	//Compute the ranges given the maxs and mins
	int i = 0;
	for (; i < numOfColumns; i++) {
		ranges[i] = 1/(maxs[i] - mins[i]);
	}
	//broadcast the ranges
	MPI_Bcast(ranges, numOfColumns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	printf("rank:%d: Ranges successfully computed..\n", 0);
}

static int setNumOfAttrs(char* line, int* numAttrs,int* strAttrs, int numOfColumns)
{
	char *token, *saveptr;
	long i = 0;
	char buff[strlen(line) + 1];
	strcpy(buff, line);
	token = strtok_r(buff, ",", &saveptr);
	int na=0,sa=0;

	while (token != NULL && i<numOfColumns) {
		if(isNumeric(token))
			na++;
		else
			sa++;
		i++;
		token = strtok_r(NULL, ",", &saveptr);
	}
	//printf("na=%d; sa=%d\n", na, sa);
	*numAttrs = na;
	*strAttrs = sa;
	//printf("done tokenizing\n");
	return 1;
}

/*
 * This function loads the partition Pk assigned to the current slave node and keeps it in memory throughout the execution of the program.
 * this function assumes that each line 400 characters or less.
 * dataPath: this is the location of the input dataset file.
 * rank: the rank of the slave node
 * rows: This is a structure to store the loaded rows.
 * chunkIndex: the row index at which the partition Pk starts
 * chunkSize: the number of rows contained in the Pk partition
 * ranges: this is the ranges vector which is used to divide by every value that gets loaded. This is an optimization step to avoid division inside a later big loop
 * numOfColumns: this is the number of condition attributes in the dataset.
*/
static int loadDataChunk(char* dataPath, int rank, Rows* rows, int numOfColumns, long chunkIndex, long chunkSize, double* ranges) {

	long size = sizeof(char) * 400 * chunkSize;
	char* buffer = malloc(size);
	//read a number of lines equal to chunkSize and starting at chunkIndex from the dataPath and store them on buffer
	readChunkFromFile(chunkIndex,chunkSize,buffer,dataPath,rank);
	char *token, *saveptr;
	long i = 0;

	token = strtok_r(buffer, "\n", &saveptr);
	int numAttrs, strAttrs;
	//Compute the number of string attributes and number of numeric attributes in the file
	setNumOfAttrs(token, &numAttrs,&strAttrs,numOfColumns);
	initRows(rows,chunkSize,strAttrs,numAttrs);
	//Go over each row, parse it and store it on "rows"
	while (token != NULL && i < chunkSize) {
		addLineToRows(rows,i,token,ranges);
		i++;
		token = strtok_r(NULL, "\n", &saveptr);
	}
	free(buffer);
	return 1;

}


/*
 * Process one row by computing partial upper and lower approx values.
 * rank: the rank of the slave node
 * rows: a structure representing the rows in the Pk partition loaded in memory of the slave
 * row: the row to be processed
 * numOfRows: this is the total number of rows in the dataset.
 * numOfColumns: this is the number of condition attributes in the dataset.
 * numOfSlaves: this is the number of slave nodes in the cluster
 * numOfClasses: this is the number of class vectors
 * numOfThreads: this is the number of threads to run in parallel per slave node
 * cvector: this is a vector to contain the classvectors. It should be initialized before being passed to this function.
 * uApprox: a vector that stores the partial upper approximation values. All such vectors at all slave nodes are meant to be reduced into one final vector.
 * lApprox: a vector that stores the partial lower approximation values. All such vectors at all slave nodes are meant to be reduced into one final vector.
 * chunkSize: the number of rows in the Pk partition loaded in memory
 * startIndex: the index of the first row in Pk
 * approxIndex: the index of the row being processed
 */
static int processFuzzyRow(Rows* localRows, Rows* row, int numOfColumns, long numOfRows, int numOfClasses, long chunkSize, long startIndex, long approxIndex, double* uApprox, double* lApprox, double* cvector, int numOfThreads,int rank) {
	long i = 0;
	Rows localRow;
	double simVal = -1;
	double uApproxVal[numOfClasses];
	initArray(uApproxVal,-1,numOfClasses);
	double lApproxVal[numOfClasses];
	initArray(lApproxVal,1,numOfClasses);
	int c=0;
	double fuzzyConj = -1;

	//for each row in Pk
	for (i = 0; i < chunkSize; i++) {
		getRowAt(localRows, i, &localRow);
		if (startIndex + i == approxIndex) //if are comparing the row with itself then the similarity value = 1
			simVal = 1;
		else //else compute the similarity value between the two rows
			simVal = getSimilarityOfRows(&localRow, row);

		//compute the upper approx and lower approx values for all classvectors
		for (c = 0; c < numOfClasses; c++) {
			fuzzyConj = fuzzyAnd(simVal, cvector[c * numOfRows + startIndex + i]);
			uApproxVal[c] = max(uApproxVal[c], fuzzyConj);
			lApproxVal[c] = min(lApproxVal[c], fuzzyImply(simVal, cvector[c * numOfRows + startIndex + i]));
		}
	}
	//update upper and lower approx values at approxIndex for all classes
	for (c = 0; c < numOfClasses; c++) {
		uApprox[c * numOfRows + approxIndex] = uApproxVal[c];
		lApprox[c * numOfRows + approxIndex] = lApproxVal[c];
	}
	return 1;
}

/*
 * Processes one line by computing its partial lower and upper approximation values
 * rank: rank of the calling slave node
 * ln: line number
 * line: line content
 * param: params to be passed to the processFuzzyRow function
 */
static int processOneLine(int rank, long ln, char* line, void* param) {

	ProcessRowArgs* args = (ProcessRowArgs*)param;
	long startIndex = args->startIndex;
	Rows currentRow = args->currentRow;
	double* ranges = args->ranges;
	Rows rows = args->rows;
	int numOfColumns = args->numOfColumns;
	long numOfRows = args->numOfRows;
	int numOfClasses = args->numOfClasses;
	long chunkSize = args->chunkSize;

	double* uApprox = args->uApprox;
	double* lApprox = args->lApprox;
	double* cvector = args->cvector;
	int numOfThreads = args->numOfThreads;

	//parse the line
	addLineToRows(&currentRow, 0, line, ranges);
	//process the line and update the uApprox and lApprox vectors
	processFuzzyRow(&rows, &currentRow, numOfColumns, numOfRows, numOfClasses, chunkSize, startIndex, ln, uApprox, lApprox, cvector, numOfThreads, rank);

	return 1;
}

/*
 * prints the progress status of the program
 */
static int printProgress(long rowsCompleted, long numOfRows, time_t* start_time, int rank) {
	time_t end_time;				// = args->end_time;
	time_t now;				// = args->now;
	time(&now);
	char percentChar = 37; //this is the percent character
	double totalTime;
	double completed = 100.0 * rowsCompleted / numOfRows;
	char hostname[1024];
	hostname[1023] = '\0';
	gethostname(hostname, 1023);
	if (rowsCompleted > 0 && rowsCompleted % 1000 == 0) {

		time(&end_time);
		totalTime = difftime(end_time, *start_time);
		printf("%s-rank%d: %.2lf%c done. Total time=%.2lf\n", hostname,rank, completed, percentChar, totalTime);
		time(&now);
	}
	return 1;

}

/*
 * This function is called by the multi-threaded loop. It processes one line.
 */
static int processOneLineP(long start, long end, long increment, int threadId, void* param)
{
	int i = 0;

	ProcessRowArgs* args = (ProcessRowArgs*) param;
	long ln = args->startLine + start;
	char* lines = args->lines;
	int* indices = args->indices;
	int rank = args->rank;
	char* line;
	long numOfRows = args->numOfRows;
	time_t start_time = *(args->start_time);

	for (i = start; i < end; i++, ln++) {
		long startIndex = args->startIndex;
		long endIndex = args->endIndex;
		if (!(ln >= startIndex && ln < endIndex)) {
			line = lines + indices[i];
			processOneLine(rank, ln, line, param);
		}
		else
		{
			Rows currentLocalRow;
			Rows rows = args->rows;
			int numOfColumns = args->numOfColumns;
			long numOfRows = args->numOfRows;
			int numOfClasses = args->numOfClasses;
			long chunkSize = args->chunkSize;

			double* uApprox = args->uApprox;
			double* lApprox = args->lApprox;
			double* cvector = args->cvector;
			int numOfThreads = args->numOfThreads;
			int rank = args->rank;

			long localRowIndex = ln - startIndex;
			getRowAt(&rows, localRowIndex, &currentLocalRow);
			processFuzzyRow(&rows, &currentLocalRow, numOfColumns, numOfRows, numOfClasses, chunkSize, startIndex, ln, uApprox, lApprox, cvector, numOfThreads, rank);

		}
		printProgress(ln,numOfRows,&start_time,rank);
	}
	return 1;
}


/*
 * This function processes a set of lines stored in "lines". It saves partial upper and lower approx values for each processed line.
 * rank: the rank of the slave node calling the function
 * startLine: This is the index of the first line in "lines" with respect to the dataset
 * lines: this contains all lines. This is meant to be used with the "indices" parameter
 * param: this is a set of arguments passed to the function.
 * numOfLines: the number of lines in "lines"
 * indices: this vector contains indices of each row in "lines" so that it is easy to access one line in "lines"
 */
static int processLines(int rank, long startLine, char* lines, void* param, int numOfLines, int* indices) {
	int i=0;
	long ln = startLine;
	char* line = lines;

	ProcessRowArgs* args = (ProcessRowArgs*)param;
	long rowsCompleted = args->rowsCompleted;
	long numOfRows = args->numOfRows;
	time_t start_time = args->start_time;

	int numOfThreads = args->numOfThreads;
	if(numOfThreads>1)
	{
		args->startLine = startLine;
		args->lines = lines;
		args->indices = indices;
		args->rank = rank;
		//divide the loops over multiple threads to run in parallel
		parallelLoop(0,numOfLines,1,args->numOfThreads,processOneLineP,param);
	}
	else
	{
		for(i=0, ln=startLine;i<numOfLines;i++, ln++)
		{
			line = lines+indices[i];
			processOneLine(rank,ln,line,param);
			rowsCompleted++;
			printProgress(rowsCompleted,numOfRows,&start_time,rank);
		}
		args->rowsCompleted = rowsCompleted;
	}
	return 1;
}

/*
 * Read all rows of a text file. Multiple lines are read at a time and passed to a function
 */
static int readMLines(long numLines, const char * fname, int (*call_back)(int, long, const char*, void*, int, int*), void* param, int rank) {
	struct stat fs;
	char *buf, *buf_end;
	char *begin, *end, c;
	char lines[1000000];
	int indices[numLines];
//printf("param=%d\n",param);
	long lineNumber = 0;
	int fd = open(fname, O_RDONLY);
	if (fd == -1) {
		err(1, "open: %s", fname);
		return 0;
	}

	if (fstat(fd, &fs) == -1) {
		err(1, "stat: %s", fname);
		return 0;
	}

	/* fs.st_size could have been 0 actually */
	buf = mmap(0, fs.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (buf == (void*) -1) {
		err(1, "mmap: %s", fname);
		close(fd);
		return 0;
	}

	buf_end = buf + fs.st_size;

	begin = end = buf;
	int linesCounter = 0;
	indices[linesCounter] = end-begin;
	int i=0;
	while (1) {
		if (!(*end == '\r' || *end == '\n')) {
			if (++end < buf_end)
				continue;
		} else if (1 + end < buf_end) {
			/* see if we got "\r\n" or "\n\r" here */
			c = *(1 + end);
			if ((c == '\r' || c == '\n') && c != *end)
				++end;
		}

		/* call the call back and check error indication. Announce
		 error here, because we didn't tell call_back the file name */

		if (begin >= buf_end || end >= buf_end) {
			//printf("final: lc=%d; range=%d; indices[i]=%d\n", linesCounter, end - begin, indices[linesCounter]);
			strncpy(lines, begin, end - begin);
			for(i=1;i<linesCounter;i++)
				lines[indices[i]-1] = '\0';
			lines[end - begin] = '\0';
			if (!call_back(rank, lineNumber, lines, param, linesCounter, indices)) {
				err(1, "[callback] %s", fname);
				break;
			}
			break;
		}
		if(linesCounter == numLines - 1)
		{
			strncpy(lines, begin, end - begin);
			for(i=1;i<numLines;i++)
				lines[indices[i]-1] = '\0';
				//printf("indices[i]=%d\n", indices[i]);
			lines[end - begin] = '\0';
			if (!call_back(rank, lineNumber, lines, param, numLines, indices)) {
				err(1, "[callback] %s", fname);
				break;
			}
			lineNumber+=numLines;
			linesCounter = 0;
			begin = ++end;
			indices[linesCounter] = end-begin;

		}
		else
		{
			end++;
			linesCounter++;
			indices[linesCounter] = end-begin;

		}

		//lineNumber++;
	}

	munmap(buf, fs.st_size);
	close(fd);
	return 1;
}

/*
 * This function runs the slaves program. It basically does everything starting with computing max and min values for the ranges.
 * Then compute upper and lower approximations in a distributed and parallel way.
 * dataPath: this is the location of the input dataset file.
 * rank: the rank of the slave node
 * numOfRows: this is the total number of rows in the dataset.
 * numOfColumns: this is the number of condition attributes in the dataset.
 * numOfSlaves: this is the number of slave nodes in the cluster
 * numOfClasses: this is the number of class vectors
 * numOfThreads: this is the number of threads to run in parallel per slave node
 * cvector: this is a vector to contain the classvectors. It should be initialized before being passed to this function.
 * uApprox: a vector that stores the partial upper approximation values. All such vectors at all slave nodes are meant to be reduced into one final vector.
 * lApprox: a vector that stores the partial lower approximation values. All such vectors at all slave nodes are meant to be reduced into one final vector.
 * rowsPerRead: this is the number of rows that are to be read from the dataset file at a time.
 * This is usually equal to the number of threads because each thread will take case of processing one row.
 */
int slaves(char* dataPath, int rank, int numOfColumns, long numOfRows, int numOfClasses, double* uApprox, double*lApprox, int numOfThreads, int numOfSlaves, long rowsPerRead, double* cvector) {
	/*
	 * Initialize variables
	 */
	double ranges[numOfColumns];
	Rows rows;
	double mins[numOfColumns];
	double maxs[numOfColumns];

	//Receive the broadcasted class vectors
	MPI_Bcast(cvector, numOfRows * numOfClasses, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/*
	 * Compute the mins and maxs of each attribute in the dataset file and store them in maxs and mins vectors
	 */
	int fd = open(dataPath, O_RDONLY);
	RangesArguments args;
	args.max = maxs;
	args.min = mins;
	args.numOfColumns = numOfColumns;
	readLines(fd, dataPath, computeRanges, &args, 0);
	close(fd);
	//end computing mins and maxs

	//Reduce the mins and maxs at the master by grouping all values computed at all slaves.
	MPI_Reduce(mins, ranges, numOfColumns, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(maxs, ranges, numOfColumns, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	//received broadcasted ranges vector from the master node
	MPI_Bcast(ranges, numOfColumns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	long chunkSize = 1 + numOfRows / numOfSlaves;
	long startIndex, endIndex;
	startIndex = (rank - 1) * chunkSize;

	if (rank == numOfSlaves)
		chunkSize = numOfRows - (chunkSize * (numOfSlaves - 1));
	endIndex = startIndex + chunkSize;

	//Load the partition Pk assigned to be process by the current slave node. This gets parsed and persisted in memory.
	loadDataChunk(dataPath,rank,&rows,numOfColumns, startIndex, chunkSize,ranges);

	time_t start_time, now;

	time(&start_time);
	time(&now);
	printf("rank%d: Starting...\n",rank);

	Rows currentRow;
	initRows(&currentRow,1,rows.numOfStringAttr,rows.numOfNumericAttr);

	int rowsCompeleted = 0;
	ProcessRowArgs rowArgs;
	rowArgs.chunkSize = chunkSize;
	rowArgs.currentRow = currentRow;
	rowArgs.cvector = cvector;
	rowArgs.lApprox = lApprox;
	rowArgs.numOfClasses = numOfClasses;
	rowArgs.numOfColumns = numOfColumns;
	rowArgs.numOfRows = numOfRows;
	rowArgs.numOfThreads = numOfThreads;
	rowArgs.ranges = ranges;
	rowArgs.rows = rows;
	rowArgs.startIndex = startIndex;
	rowArgs.endIndex = endIndex;
	rowArgs.uApprox = uApprox;
	rowArgs.rowsCompleted = rowsCompeleted;
	rowArgs.start_time = &start_time;

	/*
	 * Now this function goes over every line in the dataset file and process it by computing partial upper and lower approx values for it.
	 * Note that multiple lines are being read and processes at a time and in parallel.
	 */
	readMLines(rowsPerRead,dataPath,processLines,(void*)&rowArgs,rank);
	printf("rank%d: done with slave\n", rank);
	return 1;
}

/*
 * Store the content of upper and lower approximation vectors to disk.
 */
static int saveApproxToFile(long numOfRows, int nr_class_vectors, char *outputPath, double *upperApprox, double *lowerApprox) {
	FILE *output_file;
	int c = 0, r = 0;

	char tempName[400];
	memset(tempName, 0, 400);
	printf("classvectors = %d\n", nr_class_vectors);
	for (c = 0; c < nr_class_vectors; c++) {
		sprintf(tempName, "%s%s%d", outputPath, "output", c + 1);

		output_file = fopen(tempName, "w");
		fprintf(output_file, "upper approximation (left) and lower approximation (right) for classvector%d:\n", c + 1);

		for (r = 0; r < numOfRows; r++) {
			fprintf(output_file, "%.2f\t%.2f\n", upperApprox[r + (c * numOfRows)], lowerApprox[r + (c * numOfRows)]);
		}
		fclose(output_file);
	}
	return 0;
}

int main(int argc, char** argv) {
	/*
	 * Reading input parameters
	 */
	char* dataPath = argv[1];
	char* cvectorPath = argv[2];
	int numOfColumns = 0;
	sscanf(argv[3], "%i", &numOfColumns);
	int numOfRows = 0;
	sscanf(argv[4], "%i", &numOfRows);
	int numOfThreads = 1;
	sscanf(argv[5], "%i", &numOfThreads);
	int numOfClasses = 1;
	sscanf(argv[6], "%i", &numOfClasses);
	long rowsPerRead = 1;
		sscanf(argv[7], "%i", &rowsPerRead);
	int rank, size;
	/*
	 * Initializing result vectors and class vectors
	 */
	double* uApprox = malloc(sizeof(double) * numOfRows * numOfClasses);
	double* lApprox = malloc(sizeof(double) * numOfRows * numOfClasses);
	double* cvector = malloc(sizeof(double) * numOfRows * numOfClasses);
	initArray(uApprox, 0, numOfRows * numOfClasses);
	initArray(lApprox, 1, numOfRows * numOfClasses);
	double* finalUpperApprox = malloc(sizeof(double) * numOfRows * numOfClasses );
	double* finalLowerApprox = malloc(sizeof(double) * numOfRows * numOfClasses );
	time_t start_time, end_time;
	start_time = time(NULL);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		//running the master program
		masters(dataPath, cvectorPath, numOfRows, numOfColumns,size-1,numOfClasses, cvector);
	} else {
		//running the slaves program
		slaves(dataPath, rank, numOfColumns, numOfRows,numOfClasses, uApprox, lApprox, numOfThreads,size-1, rowsPerRead, cvector);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	//reduce the upper approximation of all slaves such that one value in the final vector will be the max of all values in the other vectors at the same index
	MPI_Reduce(uApprox, finalUpperApprox, numOfRows*numOfClasses, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//reduce the lower approximation of all slaves such that one value in the final vector will be the min of all values in the other vectors at the same index
	MPI_Reduce(lApprox, finalLowerApprox, numOfRows*numOfClasses, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	//Save final result to disk at the master
	if (rank == 0)
		saveApproxToFile(numOfRows, numOfClasses, "", finalUpperApprox, finalLowerApprox);

	MPI_Barrier(MPI_COMM_WORLD);
	end_time = time(NULL);
	if (rank == 0)
		printf("Elapsed: %lf seconds\n", difftime(end_time, start_time));
	MPI_Finalize();
	return 0;
}
