/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * https://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#include <float.h>
#include <math.h>
#include <neatogen/digcola.h>
#include <util/alloc.h>
#ifdef DIGCOLA
#include <neatogen/kkutils.h>
#include <neatogen/matrix_ops.h>
#include <neatogen/conjgrad.h>
#include <stdbool.h>

static void
standardize(double* orthog, int nvtxs) 
{
	double len, avg = 0;
    int i;
	for (i=0; i<nvtxs; i++)
		avg+=orthog[i];
	avg/=nvtxs;
	
	/* centralize: */
	for (i=0; i<nvtxs; i++)
		orthog[i]-=avg;
	
	/* normalize: */
	len = norm(orthog, nvtxs-1);

	// if we have a degenerate length, do not attempt to scale by it
	if (fabs(len) < DBL_EPSILON) {
		return;
	}

	vectors_scalar_mult(nvtxs, orthog, 1.0 / len, orthog);
}

static void
mat_mult_vec_orthog(float** mat, int dim1, int dim2, double* vec, 
    double* result, double* orthog)
{
	/* computes mat*vec, where mat is a dim1*dim2 matrix */
	int i,j;
	double sum;
	
	for (i=0; i<dim1; i++) {
		sum=0;
		for (j=0; j<dim2; j++) {
			sum += mat[i][j]*vec[j];
		}
		result[i]=sum;
	}
	assert(orthog != NULL);
	double alpha = -vectors_inner_product(dim1, result, orthog);
	scadd(result, dim1 - 1, alpha, orthog);	
}

static void
power_iteration_orthog(float** square_mat, int n, int neigs, 
     double** eigs, double* evals, double* orthog, double p_iteration_threshold)
{
  // Power-Iteration with
  //   (I - orthog × orthogᵀ) × square_mat × (I - orthog × orthogᵀ)

	int i,j;
	double *tmp_vec = gv_calloc(n, sizeof(double));
	double *last_vec = gv_calloc(n, sizeof(double));
	double *curr_vector;
	double len;
	double angle;
	double alpha;
	int largest_index;
	double largest_eval;

	double tol=1-p_iteration_threshold;

	if (neigs>=n) {
		neigs=n;
	}

	for (i=0; i<neigs; i++) {
		curr_vector = eigs[i];
		/* guess the i-th eigen vector */
choose:
		for (j=0; j<n; j++) {
			curr_vector[j] = rand()%100;
		}

		assert(orthog != NULL);
		alpha = -vectors_inner_product(n, orthog, curr_vector);
		scadd(curr_vector, n - 1, alpha, orthog);	
			// orthogonalize against higher eigenvectors
		for (j=0; j<i; j++) {
			alpha = -vectors_inner_product(n, eigs[j], curr_vector);
			scadd(curr_vector, n-1, alpha, eigs[j]);
	    }
		len = norm(curr_vector, n-1);
		if (len<1e-10) {
			/* We have chosen a vector colinear with prvious ones */
			goto choose;
		}
		vectors_scalar_mult(n, curr_vector, 1.0 / len, curr_vector);	
		do {
			copy_vector(n, curr_vector, last_vec);
			
			mat_mult_vec_orthog(square_mat,n,n,curr_vector,tmp_vec,orthog);
			copy_vector(n, tmp_vec, curr_vector);
						
			/* orthogonalize against higher eigenvectors */
			for (j=0; j<i; j++) {
				alpha = -vectors_inner_product(n, eigs[j], curr_vector);
				scadd(curr_vector, n-1, alpha, eigs[j]);
			}
			len = norm(curr_vector, n-1);
			if (len<1e-10) {
			    /* We have reached the null space (e.vec. associated 
                 * with e.val. 0)
                 */
				goto exit;
			}

			vectors_scalar_mult(n, curr_vector, 1.0 / len, curr_vector);
			angle = vectors_inner_product(n, curr_vector, last_vec);
		} while (fabs(angle)<tol);
        /* the Rayleigh quotient (up to errors due to orthogonalization):
         * u*(A*u)/||A*u||)*||A*u||, where u=last_vec, and ||u||=1
         */
		evals[i]=angle*len;
	}
exit:
	for (; i<neigs; i++) {
		/* compute the smallest eigenvector, which are 
		 * probably associated with eigenvalue 0 and for
		 * which power-iteration is dangerous
         */
		curr_vector = eigs[i];
		/* guess the i-th eigen vector */
		for (j=0; j<n; j++)
			curr_vector[j] = rand()%100;
		/* orthogonalize against higher eigenvectors */
		for (j=0; j<i; j++) {
			alpha = -vectors_inner_product(n, eigs[j], curr_vector);
			scadd(curr_vector, n-1, alpha, eigs[j]);
	    }
		len = norm(curr_vector, n-1);
		vectors_scalar_mult(n, curr_vector, 1.0 / len, curr_vector);
		evals[i]=0;
		
	}

	/* sort vectors by their evals, for overcoming possible mis-convergence: */
	for (i=0; i<neigs-1; i++) {
		largest_index=i;
		largest_eval=evals[largest_index];
		for (j=i+1; j<neigs; j++) {
			if (largest_eval<evals[j]) {
				largest_index=j;
				largest_eval=evals[largest_index];
			}
		}
		if (largest_index!=i) { // exchange eigenvectors:
			copy_vector(n, eigs[i], tmp_vec);
			copy_vector(n, eigs[largest_index], eigs[i]);
			copy_vector(n, tmp_vec, eigs[largest_index]);

			evals[largest_index]=evals[i];
			evals[i]=largest_eval;
		}
	}
	
	free (tmp_vec); free (last_vec);

}

static float* 
compute_avgs(DistType** Dij, int n, float* all_avg) 
{
	float* row_avg = gv_calloc(n, sizeof(float));
	int i,j;
	double sum=0, sum_row;

	for (i=0; i<n; i++) {
		sum_row=0;
		for (j=0; j<n; j++) {
			sum+=(double)Dij[i][j]*(double)Dij[i][j];
			sum_row+=(double)Dij[i][j]*(double)Dij[i][j];
		}
		row_avg[i]=(float)sum_row/n;
	}
	*all_avg=(float)sum/(n*n);
    return row_avg;
}

static float**
compute_Bij(DistType** Dij, int n)
{
	int i,j;
	float *storage = gv_calloc(n * n, sizeof(float));
	float **Bij = gv_calloc(n, sizeof(float *));
	float* row_avg; 
    float all_avg;

	for (i=0; i<n; i++) 
		Bij[i] = storage+i*n;

	row_avg = compute_avgs(Dij, n, &all_avg);	
	for (i=0; i<n; i++) {
		for (j=0; j<=i; j++) {
			Bij[i][j]=-(float)Dij[i][j]*Dij[i][j]+row_avg[i]+row_avg[j]-all_avg;
			Bij[j][i]=Bij[i][j];
		}
	}
    free (row_avg);
    return Bij;
}

static void
CMDS_orthog(int n, int dim, double** eigs, double tol,
            double* orthog, DistType** Dij)
{
	int i,j;
	float** Bij = compute_Bij(Dij, n);
	double *evals = gv_calloc(dim, sizeof(double));
	
	assert(orthog != NULL);
	double *orthog_aux = gv_calloc(n, sizeof(double));
	for (i=0; i<n; i++) {
		orthog_aux[i]=orthog[i];
	}
	standardize(orthog_aux,n);
    power_iteration_orthog(Bij, n, dim, eigs, evals, orthog_aux, tol);
	
	for (i=0; i<dim; i++) {
		for (j=0; j<n; j++) {
			eigs[i][j]*=sqrt(fabs(evals[i])); 
		}
	}
	free (Bij[0]); free (Bij);
	free (evals); free (orthog_aux);
}

#define SCALE_FACTOR 256

int IMDS_given_dim(vtx_data* graph, int n, double* given_coords, 
       double* new_coords, double conj_tol)
{
	int iterations2;
	int i,j, rv = 0;
	DistType** Dij;
	double* x = given_coords;	
	double uniLength;
	double* y = new_coords;
	float **lap = gv_calloc(n, sizeof(float *));
	float degree;
	double pos_i;
	double *balance = gv_calloc(n, sizeof(double));
	double b;
	bool converged;

	Dij = compute_apsp(graph, n);
	
	/* scaling up the distances to enable an 'sqrt' operation later 
     * (in case distances are integers)
     */
	for (i=0; i<n; i++)
		for (j=0; j<n; j++)
			Dij[i][j]*=SCALE_FACTOR;
	
	assert(x!=NULL);
	{
		double sum1, sum2;
	
		for (sum1=sum2=0,i=1; i<n; i++) {
			for (j=0; j<i; j++) {		
				sum1+=1.0/(Dij[i][j])*fabs(x[i]-x[j]);
				sum2+=1.0/(Dij[i][j]*Dij[i][j])*fabs(x[i]-x[j])*fabs(x[i]-x[j]);
			}
		}
		uniLength = isinf(sum2) ? 0 : sum1 / sum2;
		for (i=0; i<n; i++)
			x[i]*=uniLength;
	}

	/* smart ini: */
	CMDS_orthog(n, 1, &y, conj_tol, x, Dij);
	
	/* Compute Laplacian: */
	float *f_storage = gv_calloc(n * n, sizeof(float));
	
	for (i=0; i<n; i++) {
		lap[i]=f_storage+i*n;
		degree=0;
		for (j=0; j<n; j++) {
			if (j==i)
				continue;
			degree-=lap[i][j]=-1.0f/((float)Dij[i][j]*(float)Dij[i][j]); // w_{ij}
			
		}
		lap[i][i]=degree;
	}
	

	/* compute residual distances */
	/* if (x!=NULL)  */
    {
		double diff;
		for (i=1; i<n; i++) {
			pos_i=x[i];		
			for (j=0; j<i; j++) {
				diff=(double)Dij[i][j]*(double)Dij[i][j]-(pos_i-x[j])*(pos_i-x[j]);
				Dij[i][j]=Dij[j][i]=diff>0 ? (DistType)sqrt(diff) : 0;
			}
		}
	}
	
	/* Compute the balance vector: */
	for (i=0; i<n; i++) {
		pos_i=y[i];
		balance[i]=0;
		for (j=0; j<n; j++) {
			if (j==i)
				continue;
			if (pos_i>=y[j]) {
				balance[i]+=Dij[i][j]*(-lap[i][j]); // w_{ij}*delta_{ij}
			}
			else {
				balance[i]-=Dij[i][j]*(-lap[i][j]); // w_{ij}*delta_{ij}
			}
		}
	}

	for (converged=false,iterations2=0; iterations2<200 && !converged; iterations2++) {
		if (conjugate_gradient_f(lap, y, balance, n, conj_tol, n, true) < 0) {
		    rv = 1;
		    goto cleanup;
		}
		converged = true;
		for (i=0; i<n; i++) {
			pos_i=y[i];
			b=0;
			for (j=0; j<n; j++) {
				if (j==i)
					continue;
				if (pos_i>=y[j]) {
					b+=Dij[i][j]*(-lap[i][j]);
					
				}
				else {
					b-=Dij[i][j]*(-lap[i][j]);
					
				}
			}
			if ((b != balance[i]) && (fabs(1-b/balance[i])>1e-5)) {
				converged = false;
				balance[i]=b;
			}
		}
	}
	
	for (i = 0; !(fabs(uniLength) < DBL_EPSILON) && i < n; i++) {
		x[i] /= uniLength;
		y[i] /= uniLength;
	}
	
cleanup:

	free (Dij[0]); free (Dij);	
	free (lap[0]); free (lap);	
	free (balance);	
	return rv;
}

#endif /* DIGCOLA */

