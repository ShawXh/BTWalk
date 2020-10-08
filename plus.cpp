#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Only accept files that binary = 0.
 */

typedef float real;  // Precision of float numbers

char mine_file1[100] = "xx";
char mine_file2[100] = "xx";
char out_file[100] = "xx";

int *record1; 
int *record2;

long long dim1 = 0, dim2 = 0, dim = 0;
int save_as_binary = 0;
long long num_vertices = 0;
float *emb;


void Read()
{
    FILE *f1, *f2;

    f1 = fopen(mine_file1, "rb");
    f2 = fopen(mine_file2, "rb");
    if (f1 == NULL || f2 == NULL) {
        printf("embedding file doesn't exist\n");
        exit(1);
    }


    fscanf(f1, "%lld %lld", &num_vertices, &dim1);
    printf("File mine emb1: dim: %lld, num_vertices: %lld\n", dim1, num_vertices);
    
    fscanf(f2, "%lld %lld", &num_vertices, &dim2);
    printf("File mine emb2: dim: %lld, num_vertices: %lld\n", dim2, num_vertices);
    
    dim = dim1 + dim2;

    emb = (float *)malloc(dim * num_vertices * (long long)sizeof(float));
    record1 = (int *)malloc(num_vertices * sizeof(int));
    record2 = (int *)malloc(num_vertices * sizeof(int));
    if (emb == NULL || record1 == NULL || record2 == NULL) {
        printf("mem error\n");
        exit(1);
    }
    for (int i = 0; i < num_vertices; i++)
    {
        record1[i] = 0;
        record2[i] = 0;
    }


    printf("reading emb...\n");
    long long vid;
    float val;
    for (int i = 0; i < num_vertices; i++)
    {
        fscanf(f1, "\n%lld", &vid);
        for (int j = 0; j < dim1; j++)
        {
            fscanf(f1, " %f", &val);
            emb[vid * dim + j] = val;
        }
        record1[vid] = 1;
    }
    fclose(f1);

    for (int i = 0; i < num_vertices; i++)
    {
        fscanf(f2, "\n%lld", &vid);
        for (int j = 0; j < dim2; j++)
        {
            fscanf(f2, " %f", &val);
            emb[vid * dim + dim1 + j] = val;
        }
        record2[vid] = 1;
    }
    fclose(f2);
}

int checkEmb()
{
    int flag = 1;
    for (int i = 0; i < num_vertices; i++)
    {
        if (record1[i] != record2[i]) {
            printf("error node: %d, rcd1: %d, rcd2: %d\n", i, record1[i], record2[i]);
            flag = 0;
        }
    }
    return flag;
}

void Output()
{
    printf("Saving as binary: %d\n", save_as_binary);
    FILE *fo;

	fo = fopen(out_file, "wb");
    fprintf(fo, "%lld %lld\n", num_vertices, dim);
    
    for (int a = 0; a < num_vertices; a++)
		if (record1[a] >= 1) {
			fprintf(fo, "%d ", a);
			if (save_as_binary) for (int b = 0; b < dim; b++) fwrite(&emb[a * dim + b], sizeof(float), 1, fo);
			else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb[a * dim + b]);			    
			fprintf(fo, "\n");
		}
	fclose(fo);
}

void TrainLINE() {
    Read();

    int flag;
    flag = checkEmb();
    printf("Check result: Concanating");

    if (flag != 1) {
        printf(" failed.\n");
        exit(1);
    }
    else {
        printf(" succeded.\n");
        Output();
    }
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) 
{
    int i;
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) save_as_binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-emb1", argc, argv)) > 0) strcpy(mine_file1, argv[i + 1]);
    if ((i = ArgPos((char *)"-emb2", argc, argv)) > 0) strcpy(mine_file2, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(out_file, argv[i + 1]);

    TrainLINE();
    return 0;
}
