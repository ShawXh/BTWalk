// Author: Hao Xiong, 498967825@qq.com
// The architecture of the code follows https://github.com/tangjianpku/LINE and https://github.com/tmikolov/word2vec
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <vector>

#define MAX_STRING 100
#define SIGMOID_BOUND 6

const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;
int mi_table_size = 1000;

typedef float real;  // Precision of float numbers

char network_file[100] = "tmpfile/net.txt"; 
char embedding_file[100] = "tmpfile/emb_u";
char v_embedding_file[100] = "tmpfile/emb_v";
char degree_file[100] = "tmpfile/degree.txt";

// Basic Parameters
double NEG_SAMPLING_POWER = 0.75;
int num_threads = 40, dim = 128, num_negative = 5;
int link = 0; // 0 for single net, 2 for cross net (weak link);
int tp = 1; // 0 for undirected, 1 for directed.
int is_binary = 0;
double alpha = 0.1; // parameter for first-order
int K = 2; // max hops
float w_bs = 0.2; // bread-first search paramater
int *neg_table;
float *degree; // degree for the sum-net
long long num_memblock = 0;
long long total_samples = 100, current_sample_count = 0;
long long num_edges = 0;
real init_rho = 0.025, rho;
real *emb_vertex, *emb_context;
real *mi_table, *sigmoid_table, *exp_table;
int *edge_source_id, *edge_target_id;
double *edge_weight; 
long long *alias;
double *prob;
float dc = 0.3; // weight decay on hops, e.g. 1, 0.3, 0.09, 0.027, ...
float whop[9] = {1.0}; // the max hops is 9

// Cross-Net
char confidence_file[100] = "tmpfile/confidence.txt";
int *align;  // alignment
double *conf;  //alignment weight, index by node1
std::vector<long long> *neighbor;  // alignment
int load = 0; // 0 for not pretrained; 1 for loadinig the pretrained model
int Nm = 5; // maximum number of estimated matching nodes for each node according to the matching confidence
double beta = 0.5; // weight of alignment loss
int n1, n2;
int *neg_table1, *neg_table2;

// Parameters for align sampling
int *alias_align;
double *prob_align;
int *align_map_count;  // count probable matching nodes for each node
int *new_align_map;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

/* Read degree */
void ReadDegree()
{
//    printf("Reading degree...\n");

    degree = (float *)malloc(num_memblock * sizeof(float));
    for (int i = 0; i < num_memblock; i++) degree[i] = 0;

    FILE *fin = fopen(degree_file, "rb");
    if (fin==NULL) { 
        printf("ERROR: degree file not found!\n");
        exit(1);
    }
    int vid = 0;
    float dg = 0;
    for (int i = 0; i < num_memblock; i++) {
        fscanf(fin, "%d %f\n", &vid, &dg);
        degree[vid] = dg;
    }
    fclose(fin);
//    printf("Reading degree finished.\n");
}

/* Read confidence from confidence file */
void ReadConf()
{
    if (link == 0) { printf("Jumped Reading Confidence.\n");
        return;
    }

    printf("Reading Confidence...\n");
    // node index in sum-net space
    FILE *fin;
    char str[100];
    int vid1, vid2;
    double c;

    fin = fopen(confidence_file, "rb");
    if (fin==NULL) {
        printf("ERROR: alignment file not found!\n");
        exit(1);
    }

    int num_aligns = -1;
    while (fgets(str, sizeof(str), fin)) num_aligns++;
    fclose(fin);
    printf("Num of conf data item: %d\n", num_aligns);
    
    // malloc
    align = (int *)malloc(num_memblock * Nm * sizeof(int));
    conf = (double *)malloc(num_memblock * Nm * sizeof(double)); // indexed by net1
    align_map_count = (int *)malloc(num_memblock * sizeof(int));
    if (align == NULL || conf == NULL || align_map_count == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    // initialize
    for (int i = 0; i < num_memblock * Nm; i++) 
    {
        align[i] = -1;
        conf[i] = 0;
    }
    for (int i = 0; i < num_memblock; i++) align_map_count[i] = 0;

    // read conf
    int index;
    fin = fopen(confidence_file, "rb");
    fscanf(fin, "%d %d\n", &n1, &n2);
    printf("n1: %d, n2: %d\n", n1, n2);
    for (int i = 0; i < num_aligns; i++) {
        fscanf(fin, "%d %d %lf\n", &vid1, &vid2, &c);
        index = vid1 * Nm + align_map_count[vid1];
        conf[index] = c;
        align[index] = vid2;
        align_map_count[vid1]++;
    }
    fclose(fin);

    // check align map count
    for (int k = 0; k != num_memblock; k++)
        if (align_map_count[k] > Nm) {
            printf("Error: align map count error.\n");
            printf("node: %d; count: %d\n", k, align_map_count[k]);
            exit(1);
        }

    printf("Initializing Alignment Confidence Alignment Table...\n");
    prob_align = (double *)malloc(num_memblock * Nm * sizeof(double));
    alias_align = (int *)malloc(num_memblock * Nm *sizeof(int));
    if (prob_align == NULL || alias_align == NULL) {
        printf("Memory alloc error.\n");
        exit(1);
    }

    // build sampling map
    double sum = 0;
    int offset = 0;
    int cur_small_block, cur_large_block;
    int num_small_block, num_large_block;
    
    double *norm_prob = (double*)malloc(Nm * sizeof(double));
    int *large_block = (int *)malloc(Nm * sizeof(int));
    int *small_block = (int *)malloc(Nm * sizeof(int));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL){
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    for (int i = 0; i != num_memblock; i++)
    {
        if (align_map_count[i] == 0) continue;

        sum = 0;
        offset = i * Nm;
        // normalize the prob
        for (int j = 0; j != Nm; j++) sum += conf[offset + j];
        for (int j = 0; j != Nm; j++) norm_prob[j] = conf[offset + j] * Nm / sum;
        num_small_block = 0;
        num_large_block = 0;

        for (long long k = Nm - 1; k >= 0; k--)
        {
            if (norm_prob[k]<1)
                small_block[num_small_block++] = k;
            else
                large_block[num_large_block++] = k;
        }

        while (num_small_block && num_large_block)
        {
            cur_small_block = small_block[--num_small_block];
            cur_large_block = large_block[--num_large_block];
            prob_align[offset + cur_small_block] = norm_prob[cur_small_block];
            alias_align[offset + cur_small_block] = cur_large_block;
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
            if (norm_prob[cur_large_block] < 1)
                small_block[num_small_block++] = cur_large_block;
            else
                large_block[num_large_block++] = cur_large_block;
        }

        while (num_large_block) prob_align[offset + large_block[--num_large_block]] = 1;
        while (num_small_block) prob_align[offset + small_block[--num_small_block]] = 1;
    }
} 

/* Read sum-net from the training file */
void ReadData()
{
    FILE *fin;
    char str[300];
    double weight;
    int vid1, vid2;

    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges = -1;
    while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    printf("Number of edges: %lld          \n", num_edges);

    edge_source_id = (int *)malloc(num_edges * sizeof(int));
    edge_target_id = (int *)malloc(num_edges * sizeof(int));
    edge_weight = (double *)malloc(num_edges * sizeof(double));
    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    
    // read num_vertices and edges
    fin = fopen(network_file, "rb");
//    printf("Reading edges...\n");
    fscanf(fin ,"%lld\n",  &num_memblock);
    printf("num memblock: %lld\n", num_memblock);

    neighbor = new std::vector<long long>[num_memblock];

    for (int k = 0; k != num_edges; k++)
    {
        fscanf(fin, "%d %d %lf\n", &vid1, &vid2, &weight);
        
        edge_source_id[k] = vid1;
        edge_target_id[k] = vid2;
        edge_weight[k] = weight;
        
        neighbor[vid1].push_back(vid2);
    }
    fclose(fin);

    ReadDegree();

// check
/*
    for ( int k = 0; k != num_memblock; k++)
    {
        if ((int)degree[k] != (int)neighbor[k].size())
            printf("%d %d\n", (int)degree[k], (int)neighbor[k].size());
    }
*/
}

void ReadVector()
{
    if (load == 0) 
        return;

    printf("Reading Vectors...\n");
    
    long long vid, lv, tmp1, tmp2;
    real emb;
    FILE *fu = fopen(embedding_file, "rb");
    FILE *fv = fopen(v_embedding_file, "rb");

    if (fu == NULL || (tp == 1 && fv == NULL)) {
        printf("Vector file not found\n");
        exit(1);
    }
   
    fscanf(fu, "%lld %lld %d", &tmp1, &tmp2, &dim);
    printf("File Emb_u, dim: %d, num_memblock: %lld\n", dim, tmp1);
    if (tp == 1)
    {
        fscanf(fv, "%lld %lld %d", &tmp1, &tmp2, &dim);
        printf("File Emb_v, dim: %d, num_memblock: %lld\n", dim, tmp1);
    }

    for (long long k=0; k != num_memblock; k++)
    {
        fscanf(fu, "\n%lld", &vid);
        lv = vid * dim;
        for (int c = 0; c != dim; c++)
        {
            fscanf(fu, " %f", &emb);
            emb_vertex[lv + c] = (float)emb;
        }
        
        if (tp == 1)
        {
            fscanf(fv, "\n%lld", &vid);
            lv = vid * dim;
            for (int c = 0; c != dim; c++)
            {
                fscanf(fv, " %f", &emb);
                emb_context[lv + c] = (float)emb;
            }
        }
    }
    fclose(fu);
    if (tp == 1) fclose(fv);
}

void InitWhop()
{
    whop[0] = 1.;
    for (int i=1; i<9; i++) whop[i] = whop[i-1] * dc;
}

void InitAliasTable()
{
    printf("Initializing Alias Table...\n");
    alias = (long long *)malloc(num_edges*sizeof(long long));
    prob = (double *)malloc(num_edges*sizeof(double));
    if (alias == NULL || prob == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double*)malloc(num_edges*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) 
        norm_prob[k] = edge_weight[k] * num_edges / sum;
    
    // init alias map
    for (long long k = num_edges - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    free(norm_prob);
    free(small_block);
    free(large_block);
}

// Single-Net edge sampling
long long SampleAnEdge(double rand_value1, double rand_value2)
{
    long long k = (long long)num_edges * rand_value1;
    // k and alias[k] is index
    return rand_value2 < prob[k] ? k : alias[k];
}

// Next-hop node sampling
long long Next(double rand_value, long long vid)
{
    long long next = neighbor[vid].at((int)floor(rand_value * neighbor[vid].size()));
    return next;
}

// Cross-Net align target sampling
int SampleTarget(long long vid, double rand_value1, double rand_value2)
{
    int k = (int)Nm * rand_value1;
    int offset = vid * Nm;
    return rand_value2 < prob_align[offset + k] ? k : alias_align[offset + k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
    printf("Initializing Vectors...\n");

    long long a, b;

    a = posix_memalign((void **)&emb_vertex, dim, (long long)num_memblock * dim * sizeof(real));
    if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    if (load == 0)
        for (b = 0; b < dim; b++) for (a = 0; a < num_memblock; a++)
            emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

    if (tp == 1) // directed graph
    {
        a = posix_memalign((void **)&emb_context, dim, (long long)num_memblock * dim * sizeof(real));
        if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
        if (load == 0)
            for (b = 0; b < dim; b++) for (a = 0; a < num_memblock; a++)
                emb_context[a * dim + b] = 0;
    }

}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
    printf("Initializing Neg Table...\n");

    double sum=0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table = (int *)malloc(neg_table_size * sizeof(int));
    if (neg_table == NULL) {
        printf("neg table malloc error.\n");
        exit(1);
    }

    for (int k = 0; k != num_memblock; k++)
        sum += pow(degree[k], NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(degree[vid], NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid ++;
        }
        neg_table[k] = vid - 1;
    }

    // Cross-Net
    // Neg tabel for Net1
    sum=0; cur_sum = 0; por = 0;
    vid = 0;
    neg_table1 = (int *)malloc(neg_table_size * sizeof(int));
    if (neg_table1 == NULL) {
        printf("neg table malloc error.\n");
        exit(1);
    }

    for (int k = 0; k != n1; k++)
        sum += pow(degree[k], NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(degree[vid], NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid ++;
        }
        neg_table1[k] = vid - 1;
    }

    // Neg tabel for Net2
    sum=0; cur_sum = 0; por = 0;
    vid = n1;
    neg_table2 = (int *)malloc(neg_table_size * sizeof(int));
    if (neg_table2 == NULL) {
        printf("neg table malloc error.\n");
        exit(1);
    }

    for (int k = 0; k != n2; k++)
        sum += pow(degree[k + n1], NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(degree[vid], NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid ++;
        }
        neg_table2[k] = vid - 1;
    }
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    real x;
    sigmoid_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
    exp_table = (real *)malloc((sigmoid_table_size + 1) * sizeof(real));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        exp_table[k] = exp(x);
    }
}

void InitMiTable()
{
    real x;
    mi_table = (real *)malloc((mi_table_size + 1) * sizeof(real));
    real lb = 1.0, ub = pow(3., 1. / w_bs);
    for (int k = 0; k != mi_table_size; k++)
    {
        x = (ub - lb) * k / mi_table_size + lb;
        mi_table[k] = pow(x, w_bs);
    }
}

real FastSigmoid(real x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

real FastLog(real x)
{
    if (x < 2) return 1.;
    else if (x < 4) return 2.;
    else return 3.;
}

real FastExp(real x)
{
    if (x > SIGMOID_BOUND) return exp_table[sigmoid_table_size-1];
    else if (x < -SIGMOID_BOUND) return exp_table[0];
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return exp_table[k];
}

real FastMi(real x)
{
    real lb = 1.0, ub = pow(3., 1. / w_bs);
    if (x > ub) return 3.;
    else if (x < lb) return 1.;
    real k = (x - lb) * mi_table_size / (ub - lb);
    return mi_table[(int)k];
}


/* Fastly generate a random integer */
int Rand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

/* Fastly generate a random double */
double fastRand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return ((seed >> 16) % neg_table_size) / (double)neg_table_size;
}

// First-Order Regularizer
void UpdateFirst(real *vec_u, real *vec_v, real *vec_error, double w, int mode)
{
    if (vec_u == vec_v) return;

    real g;
    for (int c = 0; c != dim; c++) {
        g = w * (vec_u[c] - vec_v[c]) * rho;
        vec_error[c] += -g;
        vec_v[c] += g;
    }
    if (mode == 0) 
        for (int c = 0; c != dim; c++) {
            vec_u[c] += vec_error[c];
            vec_error[c] = 0;
        }
}

void UpdateNeg(real *vec_u, real *vec_v)
{
    for (int c = 0; c != dim; c++)
        vec_v[c] += 0.01 * (vec_v[c] - vec_u[c]) * rho;
}

void UpdateNegFirst(int vid, unsigned long long &seed)
{
    int target;

    if (vid < n1) target = neg_table2[Rand(seed)];
    else target = neg_table1[Rand(seed)];

    int lu = vid * dim, lv = target * dim;
    UpdateNeg(&emb_vertex[lu], &emb_vertex[lv]);
    UpdateNeg(&emb_context[lu], &emb_context[lv]);
}

void UpdateAnchorFirst(int vid, real *vec_error_u, real* vec_error_t, int mode, unsigned long long &seed)
{
    double r1, r2;
    r1 = fastRand(seed);
    r2 = fastRand(seed);
    int tt = SampleTarget(vid, r1, r2);
    int target = align[vid * Nm + tt];
    
    int lu = vid * dim, lv = target * dim;
    double c = conf[vid * Nm + tt];
    double w = beta * c * c;
    if (mode == 1)
        UpdateFirst(&emb_vertex[lu], &emb_vertex[lv], vec_error_u, w, 1);
    else if (mode == 0)
        UpdateFirst(&emb_vertex[lu], &emb_vertex[lv], vec_error_t, w, 0);

    if (tp == 1) 
        UpdateFirst(&emb_context[lu], &emb_context[lv], vec_error_t, w, 0);
}

/* Update embeddings */
real Update(real *vec_u, real *vec_v, real *vec_error, int label, long long vid, unsigned long long &seed, float wh)
{
    real x = 0, g, score;
    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    score = FastSigmoid(x);
    //if (label == 0 && (score < 0.1 || score > 0.8)) return -10000.0;

    g = (label - score) * rho * wh;
    
    if (vec_u == vec_v) 
    {
        // only second-order
        if (tp == 0) g = g * 2;
        for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
        if (tp == 1) for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
    }
    else
    {
        // first-order
        if (label == 1) UpdateFirst(vec_u, vec_v, vec_error, alpha * wh, 1);
        // second-order
        for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
        for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
    }
    return x;
}

void *TrainLINEThread(void *id)
{
    long long u, v, cnode=0, lu, lv, target=0, label=0;
    long long count = 0, last_count = 0, curedge;
    unsigned long long seed = (long long)id;
    float at1 = 1., atk = 1., ip=0., sn=1.;
    int d = 0;

    real *vec_error_u = (real *)calloc(dim, sizeof(real));
    real *vec_error_v = (real *)calloc(dim, sizeof(real));
    real *vec_error_t = (real *)calloc(dim, sizeof(real));
    for (int c = 0; c != dim; c++) vec_error_t[c] = 0;

    if (vec_error_u == NULL || vec_error_v == NULL || vec_error_t == NULL) 
    {
        printf("Memory alloc error\n");
        exit(1);
    }

    while (1)
    {
        //judge for exit
        if (count > total_samples / num_threads + 2) break;

        if (count - last_count > 10000)
        {
            current_sample_count += count - last_count;
            last_count = count;
            printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (real)current_sample_count / (real)(total_samples + 1) * 100);
            fflush(stdout);
            rho = init_rho * (1 - current_sample_count / (real)(total_samples + 1));
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }

        curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        u = edge_source_id[curedge];
        v = edge_target_id[curedge];

        lu = u * dim;

        for (int k = 0; k < K; k++) 
        {
            int n = 0;
            sn = 1.;
            while(n < sn) 
            {
                // Initialize
                for (int c = 0; c != dim; c++) vec_error_u[c] = 0;
                if (tp == 1 && link == 2) for (int c = 0; c != dim; c++) vec_error_v[c] = 0;
                // Negtive Sampling
                ip = 0.;
                d = 0;
                while (d < num_negative + 1)
                {
                    if (d == 0 && k == 0)
                    {
                        target = v;
                        cnode = v;
                        label = 1;
                    }
                    else if (d == 0 && k != 0)
                    {
                        target = Next(fastRand(seed), cnode);
                        if (n == 0) cnode = target;
                        label = 1;
                    }
                    else
                    {
                        target = neg_table[Rand(seed)];
                        label = 0;
                    }
                    lv = target * dim;
                    if (d == 0)
                    {
                        if (tp == 1) ip = Update(&emb_vertex[lu], &emb_context[lv], vec_error_u, label, u, seed, whop[k]);
                        else if (tp == 0) ip = Update(&emb_vertex[lu], &emb_vertex[lv], vec_error_u, label, u, seed, whop[k]);
                    }else{
                        if (tp == 1) ip = Update(&emb_vertex[lu], &emb_context[lv], vec_error_u, label, u, seed, 1.);
                        else if (tp == 0) ip = Update(&emb_vertex[lu], &emb_vertex[lv], vec_error_u, label, u, seed, 1.);
                    }
                    // attention
                    if (d == 0) {
                        if( k == 0) {
                            at1 = FastExp(ip) * degree[v];
                        }else{ 
                            atk = FastExp(ip) * degree[target];
                            sn = (sn * n + FastMi(at1/atk))/(n + 1);
                        }
                        //printf("%d,%d,%.1f,%.1f ", n, k, FastMi(at1/atk), sn);
                    }
                    d++;
                    if (d > 1 && ip < -9999) d -= 1;
                }
                // Cross-Net Regularizor
                if (link == 2)
                {
                    if (align_map_count[u] > 0) 
                        UpdateAnchorFirst(u, vec_error_u, vec_error_t, 1, seed);
                    if (align_map_count[cnode] > 0) 
                        UpdateAnchorFirst(cnode, vec_error_u, vec_error_t, 0, seed);
                }
                // Update emb_vertex[lu]
                for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error_u[c];

                n++;
            }
        }
        count++;
    }
    free(vec_error_u);
    free(vec_error_v);
    pthread_exit(NULL);
}

void Output()
{
    printf("Writiing Embedding...\n");
    
    FILE *fo;

    // writing embedding file
    fo = fopen(embedding_file, "wb");
    fprintf(fo, "%lld %d\n", num_memblock, dim);
    for (int a = 0; a < num_memblock; a++)
        if (degree[a] >= 1) {
            fprintf(fo, "%d ", a);
            if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(float), 1, fo);
            else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);                
            fprintf(fo, "\n");
        }
    fclose(fo);
   
    if (tp == 1) { // directed graph
        // writing hidden embedding file
        fo = fopen(v_embedding_file, "wb");
        fprintf(fo, "%lld %d\n", num_memblock, dim);
        for (int a = 0; a < num_memblock; a++)
            if (degree[a] >= 1) {
                fprintf(fo, "%d ", a);
                if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_context[a * dim + b], sizeof(float), 1, fo);
                else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_context[a * dim + b]);
                fprintf(fo, "\n");
            }
        fclose(fo);
    }
}

void PrintParameters()
{
    printf("--------------------------------\n");
    printf("File Settings:  \n");
    printf("Net:\t%s \n", network_file);
    printf("Emb-u:\t%s \n", embedding_file);
    if (tp == 1) printf("Emb-v:\t%s \n", v_embedding_file);
    printf("Dgr:\t%s \n", degree_file);
    if (link == 2)
        printf("Cfd:\t%s \n", confidence_file);
    printf("--------------------------------\n");
    printf("Samples: %lldM\n", total_samples / 1000000);
    // printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("Save as binary: %d\n", is_binary);
    // printf("Neg sampling power: %f\n", NEG_SAMPLING_POWER);
    printf("alpha: %.3f\n", alpha);
    printf("decay: %.3f\n", dc);
    printf("w: %.3f\n", w_bs);
    if (link == 2) printf("beta: %.3f\n", beta);
    printf("Max-hops: %d\n", K);
    if (link == 0) printf("Link: Single-net\n");
    else if (link == 2){ 
        printf("Link: Cross-net\n");
        printf("Nm: %d  \n", Nm);
    }
    if (tp == 0) printf("Undirected Info Modeling...\n");
    else if (tp == 1) printf("Directed Info Modeling...\n");
    printf("--------------------------------\n");
}

void TrainLINE() 
{
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    PrintParameters();

    ReadData();
    ReadConf();
    InitAliasTable();
    InitVector();
    ReadVector();
    InitNegTable();
    InitSigmoidTable();
    InitMiTable();
    InitWhop();

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);

    clock_t start = clock();
    printf("--------------------------------\n");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf s\n", (double)(finish - start) / CLOCKS_PER_SEC / num_threads);

    Output();
}

int ArgPos(char *str, int argc, char **argv)
{
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
    printf("Parameters: \n");
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-decay", argc, argv)) > 0) dc = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-w", argc, argv)) > 0) w_bs = atof(argv[i + 1]);

    if ((i = ArgPos((char *)"-type", argc, argv)) > 0) tp = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-link", argc, argv)) > 0) link = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-power", argc, argv)) > 0) NEG_SAMPLING_POWER = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-load", argc, argv)) > 0) load = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-Nm", argc, argv)) > 0) Nm = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-K", argc, argv)) > 0) K = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    if ((i = ArgPos((char *)"-conf", argc, argv)) > 0) strcpy(confidence_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-degree", argc, argv)) > 0) strcpy(degree_file, argv[i + 1]);

    if ((i = ArgPos((char *)"-emb-u", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-emb-v", argc, argv)) > 0) strcpy(v_embedding_file, argv[i + 1]);


    total_samples *= 1000000;
    rho = init_rho;
    TrainLINE();
    return 0;
}
