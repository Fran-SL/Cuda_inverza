/* Autores: Francisco Soto Lagos, Sebastian Salinas Jorquera
 * Implementación completamente paralela con cálculo de speedup
 * 
 * SPEEDUP:
 * 1. Modificamos la constante TIEMPO_SECUENCIAL_MS con el tiempo secuencial medido
 * 2. El programa calcula automáticamente: speedup = T_secuencial / T_paralelo
 * 3. Los resultados se guardan en metrica.met con el speedup calculado
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <windows.h>

// Constantes y configuraciones
#define EPSILON 1e-12
#define MAX_PRECISION 15
#define NUM_ENSAYOS 10
#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

// Tiempo secuencial de referencia (modificar según tu medición)
#define TIEMPO_SECUENCIAL_MS 0.000001  // Cambiar por tu tiempo secuencial medido

// CUDA ya tiene min() definido, no necesitamos redefinirlo

// Funciones utilitarias
double obtener_tiempo_ms() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart * 1000.0;
}

void leer_matriz(const char* nombre_archivo, double** matriz_destino, int* filas, int* columnas) {
    if (!nombre_archivo || !matriz_destino || !filas || !columnas) {
        printf(" ERROR: Parámetros inválidos para lectura de matriz\n");
        exit(1);
    }
    
    FILE* archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        printf(" ERROR: No se pudo abrir el archivo %s\n", nombre_archivo);
        exit(1);
    }

    if (fscanf(archivo, "%d %d", filas, columnas) != 2) {
        printf(" ERROR: Formato incorrecto en dimensiones del archivo de entrada\n");
        fclose(archivo);
        exit(1);
    }
    
    if (*filas <= 0 || *columnas <= 0) {
        printf(" ERROR: Dimensiones inválidas: %dx%d\n", *filas, *columnas);
        fclose(archivo);
        exit(1);
    }
    
    const int total_elementos = (*filas) * (*columnas);
    const size_t tamano_memoria = total_elementos * sizeof(double);

    *matriz_destino = (double*)malloc(tamano_memoria);
    if (!*matriz_destino) {
        printf(" ERROR: No se pudo reservar memoria para matriz %dx%d (%zu bytes)\n", 
               *filas, *columnas, tamano_memoria);
        fclose(archivo);
        exit(1);
    }

    for (int indice_elemento = 0; indice_elemento < total_elementos; indice_elemento++) {
        if (fscanf(archivo, "%lf", &(*matriz_destino)[indice_elemento]) != 1) {
            printf(" ERROR: Datos insuficientes en archivo (elemento %d/%d)\n", 
                   indice_elemento + 1, total_elementos);
            free(*matriz_destino);
            *matriz_destino = NULL;
            fclose(archivo);
            exit(1);
        }
    }

    fclose(archivo);
    printf("  Matriz %dx%d leída exitosamente (%d elementos)\n", 
           *filas, *columnas, total_elementos);
}

/**
 * Función optimizada para guardar la pseudoinversa en archivo de salida
 * 
 * Formato del archivo salida.sal:
 * Línea 1: tipo de pseudoinversa ('L' o 'R')
 * Líneas siguientes: elementos de la pseudoinversa con alta precisión
 * 
 * Parámetros:
 *   - pseudoinversa: matriz calculada
 *   - filas, columnas: dimensiones de la pseudoinversa  
 *   - tipo_pseudoinversa: 'L' para izquierda, 'R' para derecha
 */
void guardar_pseudoinversa(double* pseudoinversa, int filas, int columnas, char tipo_pseudoinversa) {
    // Validación de parámetros de entrada
    if (!pseudoinversa || filas <= 0 || columnas <= 0) {
        printf(" ERROR: Parámetros inválidos para guardar pseudoinversa\n");
        return;
    }
    
    if (tipo_pseudoinversa != 'L' && tipo_pseudoinversa != 'R') {
        printf(" ERROR: Tipo de pseudoinversa inválido: %c (debe ser 'L' o 'R')\n", tipo_pseudoinversa);
        return;
    }
    
    FILE* archivo_salida = fopen("salida.sal", "w");
    if (!archivo_salida) {
        printf(" ERROR: No se pudo crear el archivo salida.sal\n");
        return;
    }
    
    // Escribir tipo de pseudoinversa
    fprintf(archivo_salida, "%c\n", tipo_pseudoinversa);
    
    // Optimización: Calcular total de elementos
    const int total_elementos = filas * columnas;
    
    // Escribir matriz con alta precisión de forma optimizada
    for (int fila = 0; fila < filas; fila++) {
        const int offset_fila = fila * columnas;
        
        for (int columna = 0; columna < columnas; columna++) {
            if (columna > 0) fprintf(archivo_salida, " ");
            fprintf(archivo_salida, "%.15f", pseudoinversa[offset_fila + columna]);
        }
        fprintf(archivo_salida, "\n");
    }
    
    fclose(archivo_salida);
    printf("  Pseudoinversa %dx%d (tipo %c) guardada en salida.sal\n", 
           filas, columnas, tipo_pseudoinversa);
}

/**
 * Función para guardar métricas de speedup y optimización CUDA
 * Formato del archivo metrica.met:
 * Encabezados y métricas de speedup con formato tabular
 */
void guardar_metricas_speedup(double tiempo_secuencial, double tiempo_paralelo_mejor, 
                             double* tiempos_medidos, int* configuraciones_bloques, 
                             int* configuraciones_hilos, int total_ensayos) {
    if (!tiempos_medidos || !configuraciones_bloques || !configuraciones_hilos || total_ensayos <= 0) {
        printf(" ERROR: Parámetros inválidos para guardar métricas\n");
        return;
    }
    
    FILE* archivo_metricas = fopen("metrica.met", "w");
    if (!archivo_metricas) {
        printf(" ERROR: No se pudo crear el archivo metrica.met\n");
        return;
    }
    
    // Calcular speedup global
    double speedup_global = (tiempo_paralelo_mejor > 0) ? (tiempo_secuencial / tiempo_paralelo_mejor) : 0.0;
    
    // Escribir encabezado del archivo
    fprintf(archivo_metricas, "=== METRICAS DE SPEEDUP CUDA ===\n");
    fprintf(archivo_metricas, "Tiempo_Secuencial_ms: %.6f\n", tiempo_secuencial);
    fprintf(archivo_metricas, "Tiempo_Paralelo_Mejor_ms: %.6f\n", tiempo_paralelo_mejor);
    fprintf(archivo_metricas, "Speedup_Global: %.6f\n", speedup_global);
    fprintf(archivo_metricas, "\n");
    
    // Escribir encabezados de la tabla
    fprintf(archivo_metricas, "%-8s %-12s %-18s %-15s %-12s\n", 
            "Ensayo", "Bloques", "Hilos_por_Bloque", "Tiempo_ms", "Speedup");
    fprintf(archivo_metricas, "%-8s %-12s %-18s %-15s %-12s\n", 
            "------", "-------", "----------------", "---------", "-------");
    
    // Escribir métricas de cada ensayo con speedup individual
    for (int ensayo = 0; ensayo < total_ensayos; ensayo++) {
        double speedup_individual = (tiempos_medidos[ensayo] > 0) ? 
                                   (tiempo_secuencial / tiempos_medidos[ensayo]) : 0.0;
        
        fprintf(archivo_metricas, "%-8d %-12d %-18d %-15.6f %-12.6f\n", 
                ensayo + 1, 
                configuraciones_bloques[ensayo], 
                configuraciones_hilos[ensayo], 
                tiempos_medidos[ensayo], 
                speedup_individual);
    }
    
    fclose(archivo_metricas);
    printf("  Métricas con speedup %.2fx guardadas en metrica.met\n", speedup_global);
}

// Función para guardar resultado cuando no hay pseudoinversa
void guardar_sin_pseudoinversa() {
    FILE* archivo_salida = fopen("salida.sal", "w");
    if (!archivo_salida) {
        printf(" ERROR: No se pudo crear archivo salida.sal\n");
        return;
    }
    
    fprintf(archivo_salida, "-1\n");
    fclose(archivo_salida);
    printf("  Resultado 'sin pseudoinversa' guardado en salida.sal\n");
}

// ===================================================================
// KERNELS CUDA PARALELOS PARA ÁLGEBRA LINEAL
// ===================================================================

/**
 * KERNEL CUDA: Encontrar fila con el pivote más grande en una columna
 */
__global__ void kernel_find_max_pivot(double* matriz, int filas, int columnas, int col, 
                                     int start_row, double* max_values, int* max_indices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = tid + start_row;
    
    if (fila < filas) {
        double valor = fabs(matriz[fila * columnas + col]);
        max_values[tid] = valor;
        max_indices[tid] = fila;
    } else {
        max_values[tid] = 0.0;
        max_indices[tid] = -1;
    }
}

/**
 * KERNEL CUDA: Intercambiar dos filas de la matriz
 */
__global__ void kernel_swap_rows(double* matriz, int columnas, int fila1, int fila2) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < columnas && fila1 != fila2) {
        double temp = matriz[fila1 * columnas + col];
        matriz[fila1 * columnas + col] = matriz[fila2 * columnas + col];
        matriz[fila2 * columnas + col] = temp;
    }
}

/**
 * KERNEL CUDA: Eliminación gaussiana paralela para cada fila
 */
__global__ void kernel_eliminacion_gaussiana_rango(double* matriz, int filas, int columnas, 
                                                  int pivot_row, int columna_actual) {
    int fila = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (fila < filas && col < columnas && fila != pivot_row && fila > pivot_row) {
        double pivot = matriz[pivot_row * columnas + columna_actual];
        if (fabs(pivot) > EPSILON) {
            double factor = matriz[fila * columnas + columna_actual] / pivot;
            matriz[fila * columnas + col] -= factor * matriz[pivot_row * columnas + col];
        }
    }
}

/**
 * FUNCIÓN CUDA: Calcular rango de matriz completamente en paralelo
 * Implementa eliminación gaussiana con pivoteo parcial para determinar el rango
 */
int calcular_rango_cuda(double* matriz_host, int filas, int columnas) {
    if (!matriz_host || filas <= 0 || columnas <= 0) return 0;
    
    size_t size = filas * columnas * sizeof(double);
    double* gpu_matriz;
    double* gpu_max_values;
    int* gpu_max_indices;
    
    // Reservar memoria GPU
    if (cudaMalloc(&gpu_matriz, size) != cudaSuccess ||
        cudaMalloc(&gpu_max_values, filas * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&gpu_max_indices, filas * sizeof(int)) != cudaSuccess) {
        cudaFree(gpu_matriz); 
        cudaFree(gpu_max_values); 
        cudaFree(gpu_max_indices);
        return 0;
    }
    
    // Copiar datos a GPU
    cudaMemcpy(gpu_matriz, matriz_host, size, cudaMemcpyHostToDevice);
    
    int rango_actual = 0;
    int min_dim = (filas < columnas) ? filas : columnas;
    
    // Variables para CPU
    double* max_values = (double*)malloc(filas * sizeof(double));
    int* max_indices = (int*)malloc(filas * sizeof(int));
    
    // Procesamiento paralelo por columnas
    for (int col = 0; col < min_dim; col++) {
        // Configurar kernels
        const int threads_1d = min(256, filas - col);
        dim3 block(threads_1d);
        dim3 grid((filas - col + block.x - 1) / block.x);
        
        // Encontrar fila con el pivote más grande
        kernel_find_max_pivot<<<grid, block>>>(gpu_matriz, filas, columnas, col, col, 
                                               gpu_max_values, gpu_max_indices);
        cudaDeviceSynchronize();
        
        // Copiar resultados a CPU para encontrar el máximo global
        cudaMemcpy(max_values, gpu_max_values, (filas - col) * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(max_indices, gpu_max_indices, (filas - col) * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Encontrar el pivote más grande
        double max_pivot = 0.0;
        int pivot_row = -1;
        for (int i = 0; i < filas - col; i++) {
            if (max_values[i] > max_pivot) {
                max_pivot = max_values[i];
                pivot_row = max_indices[i];
            }
        }
        
        // Verificar si el pivote es válido
        if (pivot_row >= 0 && max_pivot > EPSILON) {
            // Intercambiar filas si es necesario
            if (pivot_row != col) {
                dim3 swap_block(min(256, columnas));
                dim3 swap_grid((columnas + swap_block.x - 1) / swap_block.x);
                kernel_swap_rows<<<swap_grid, swap_block>>>(gpu_matriz, columnas, col, pivot_row);
                cudaDeviceSynchronize();
            }
            
            // Hacer eliminación gaussiana
            dim3 block2(16, 16);
            dim3 grid2((filas + block2.x - 1) / block2.x, (columnas + block2.y - 1) / block2.y);
            
            kernel_eliminacion_gaussiana_rango<<<grid2, block2>>>(gpu_matriz, filas, columnas, col, col);
            cudaDeviceSynchronize();
            
            rango_actual++;
        } else {
            // No hay más pivotes válidos
            break;
        }
    }
    
    // Limpiar memoria
    cudaFree(gpu_matriz);
    cudaFree(gpu_max_values);
    cudaFree(gpu_max_indices);
    free(max_values);
    free(max_indices);
    
    return rango_actual;
}

/**
 * KERNEL CUDA: Encontrar pivote para descomposición LU
 */
__global__ void kernel_find_pivot(double* matriz, int* permutaciones, int n, int paso, int* pivot_row, double* pivot_value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = tid + paso;
    
    if (fila < n) {
        double valor = fabs(matriz[permutaciones[fila] * n + paso]);
        
        // Reducción paralela para encontrar el mejor pivote
        __shared__ double max_vals[256];
        __shared__ int max_indices[256];
        
        int local_id = threadIdx.x;
        if (local_id < blockDim.x) {
            max_vals[local_id] = valor;
            max_indices[local_id] = fila;
        }
        
        __syncthreads();
        
        // Reducción en memoria compartida
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (local_id < s && local_id + s < blockDim.x) {
                if (max_vals[local_id + s] > max_vals[local_id]) {
                    max_vals[local_id] = max_vals[local_id + s];
                    max_indices[local_id] = max_indices[local_id + s];
                }
            }
            __syncthreads();
        }
        
        // El thread 0 actualiza el pivote global
        if (local_id == 0) {
            *pivot_row = max_indices[0];
            *pivot_value = max_vals[0];
        }
    }
}

/**
 * KERNEL CUDA: Descomposición LU con pivoteo parcial (MÁS EFICIENTE)
 */
__global__ void kernel_lu_decomposition_step(double* matriz, int* permutaciones, int n, int paso) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = tid + paso + 1;
    
    if (fila < n) {
        // Obtener elemento pivote
        double pivot = matriz[permutaciones[paso] * n + paso];
        
        if (fabs(pivot) > EPSILON) {
            // Calcular factor de eliminación
            double factor = matriz[permutaciones[fila] * n + paso] / pivot;
            
            // Actualizar fila completa
            for (int col = paso + 1; col < n; col++) {
                matriz[permutaciones[fila] * n + col] -= factor * matriz[permutaciones[paso] * n + col];
            }
            
            // Guardar factor en L
            matriz[permutaciones[fila] * n + paso] = factor;
        }
    }
}

/**
 * KERNEL CUDA: Forward substitution paralela (Ly = Pb)
 */
__global__ void kernel_forward_substitution(double* L, int* permutaciones, double* b, double* y, int n, int col_b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        double suma = 0.0;
        
        // Calcular suma de elementos anteriores
        for (int j = 0; j < tid; j++) {
            suma += L[permutaciones[tid] * n + j] * y[j * n + col_b];
        }
        
        // Resolver para y[tid]
        y[tid * n + col_b] = b[permutaciones[tid] * n + col_b] - suma;
    }
}

/**
 * KERNEL CUDA: Backward substitution paralela (Ux = y)
 */
__global__ void kernel_backward_substitution(double* U, double* y, double* x, int n, int col_b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = n - 1 - tid;
    
    if (fila >= 0) {
        double suma = 0.0;
        
        // Calcular suma de elementos posteriores
        for (int j = fila + 1; j < n; j++) {
            suma += U[fila * n + j] * x[j * n + col_b];
        }
        
        // Resolver para x[fila]
        double diagonal = U[fila * n + fila];
        if (fabs(diagonal) > EPSILON) {
            x[fila * n + col_b] = (y[fila * n + col_b] - suma) / diagonal;
        }
    }
}

/**
 * FUNCIÓN CUDA: Inversión LU optimizada con máxima estabilidad
 * Implementación única con pivoteo parcial y resolución de sistemas paralela
 */
double* invertir_matriz_lu_cuda(double* matriz_host, int n) {
    if (!matriz_host || n <= 0) return NULL;
    
    size_t size = n * n * sizeof(double);
    double* gpu_matriz;
    double* gpu_identidad;
    double* gpu_resultado;
    double* gpu_temp_y;
    int* gpu_permutaciones;
    int* gpu_pivot_row;
    double* gpu_pivot_value;
    
    // Reservar memoria GPU
    if (cudaMalloc(&gpu_matriz, size) != cudaSuccess ||
        cudaMalloc(&gpu_identidad, size) != cudaSuccess ||
        cudaMalloc(&gpu_resultado, size) != cudaSuccess ||
        cudaMalloc(&gpu_temp_y, size) != cudaSuccess ||
        cudaMalloc(&gpu_permutaciones, n * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&gpu_pivot_row, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&gpu_pivot_value, sizeof(double)) != cudaSuccess) {
        
        cudaFree(gpu_matriz); cudaFree(gpu_identidad); cudaFree(gpu_resultado);
        cudaFree(gpu_temp_y); cudaFree(gpu_permutaciones); cudaFree(gpu_pivot_row);
        cudaFree(gpu_pivot_value);
        return NULL;
    }
    
    // Copiar datos a GPU
    cudaMemcpy(gpu_matriz, matriz_host, size, cudaMemcpyHostToDevice);
    
    // Crear matriz identidad en GPU
    double* host_identidad = (double*)calloc(n * n, sizeof(double));
    for (int i = 0; i < n; i++) host_identidad[i * n + i] = 1.0;
    cudaMemcpy(gpu_identidad, host_identidad, size, cudaMemcpyHostToDevice);
    
    // Inicializar permutaciones
    int* host_perm = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) host_perm[i] = i;
    cudaMemcpy(gpu_permutaciones, host_perm, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // ===== FASE 1: DESCOMPOSICIÓN LU CON PIVOTEO =====
    for (int paso = 0; paso < n - 1; paso++) {
        // Encontrar pivote óptimo
        dim3 block_pivot(min(256, n - paso));
        dim3 grid_pivot(1);
        
        kernel_find_pivot<<<grid_pivot, block_pivot>>>(gpu_matriz, gpu_permutaciones, n, paso, gpu_pivot_row, gpu_pivot_value);
        cudaDeviceSynchronize();
        
        // Intercambiar filas si es necesario (en permutaciones)
        int pivot_row_host;
        cudaMemcpy(&pivot_row_host, gpu_pivot_row, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (pivot_row_host != paso) {
            // Intercambiar permutaciones
            int temp = host_perm[paso];
            host_perm[paso] = host_perm[pivot_row_host];
            host_perm[pivot_row_host] = temp;
            cudaMemcpy(gpu_permutaciones, host_perm, n * sizeof(int), cudaMemcpyHostToDevice);
        }
        
        // Eliminación gaussiana
        if (n - paso - 1 > 0) {
            dim3 block_lu(min(256, n - paso - 1));
            dim3 grid_lu((n - paso - 1 + block_lu.x - 1) / block_lu.x);
            
            kernel_lu_decomposition_step<<<grid_lu, block_lu>>>(gpu_matriz, gpu_permutaciones, n, paso);
            cudaDeviceSynchronize();
        }
    }
    
    // ===== FASE 2: RESOLVER SISTEMAS A*X = I =====
    // Para cada columna de la matriz identidad
    for (int col = 0; col < n; col++) {
        // Forward substitution: L*y = P*e_col
        for (int fila = 0; fila < n; fila++) {
            dim3 block_forward(1);
            dim3 grid_forward(1);
            
            kernel_forward_substitution<<<grid_forward, block_forward>>>(gpu_matriz, gpu_permutaciones, gpu_identidad, gpu_temp_y, fila + 1, col);
            cudaDeviceSynchronize();
        }
        
        // Backward substitution: U*x = y
        for (int fila = n - 1; fila >= 0; fila--) {
            dim3 block_backward(1);
            dim3 grid_backward(1);
            
            kernel_backward_substitution<<<grid_backward, block_backward>>>(gpu_matriz, gpu_temp_y, gpu_resultado, n - fila, col);
            cudaDeviceSynchronize();
        }
    }
    
    // Copiar resultado final
    double* resultado = (double*)malloc(size);
    if (resultado) {
        cudaMemcpy(resultado, gpu_resultado, size, cudaMemcpyDeviceToHost);
    }
    
    // Limpiar memoria
    cudaFree(gpu_matriz); cudaFree(gpu_identidad); cudaFree(gpu_resultado);
    cudaFree(gpu_temp_y); cudaFree(gpu_permutaciones); cudaFree(gpu_pivot_row);
    cudaFree(gpu_pivot_value);
    free(host_identidad); free(host_perm);
    
    return resultado;
}

// ===================================================================
// KERNELS CUDA PARA PARALELIZACIÓN DE MATRICES
// ===================================================================

/**
 * KERNEL CUDA OPTIMIZADO: Transponer matriz en paralelo
 */
__global__ void kernel_transponer(double* matriz_origen, double* matriz_transpuesta, 
                                  int filas_origen, int columnas_origen) {
    const int columna_global = blockIdx.x * blockDim.x + threadIdx.x;
    const int fila_global = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (columna_global < columnas_origen && fila_global < filas_origen) {
        const int indice_origen = fila_global * columnas_origen + columna_global;
        const int indice_transpuesta = columna_global * filas_origen + fila_global;
        matriz_transpuesta[indice_transpuesta] = matriz_origen[indice_origen];
    }
}

/**
 * KERNEL CUDA OPTIMIZADO: Multiplicar matrices en paralelo
 */
__global__ void kernel_multiplicar(double* matriz_A, double* matriz_B, double* matriz_C, 
                                   int filas_A, int columnas_A, int columnas_B) {
    const int fila_resultado = blockIdx.y * blockDim.y + threadIdx.y;
    const int columna_resultado = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fila_resultado < filas_A && columna_resultado < columnas_B) {
        double acumulador_suma = 0.0;
        const int offset_fila_A = fila_resultado * columnas_A;
        
        for (int k = 0; k < columnas_A; k++) {
            const double elemento_A = matriz_A[offset_fila_A + k];
            const double elemento_B = matriz_B[k * columnas_B + columna_resultado];
            acumulador_suma += elemento_A * elemento_B;
        }
        
        const int indice_resultado = fila_resultado * columnas_B + columna_resultado;
        matriz_C[indice_resultado] = acumulador_suma;
    }
}

/**
 * FUNCIÓN CUDA PARALELA: Calcular pseudoinversa usando algoritmo LU
 */
double* calcular_pseudoinversa_cuda_paralela(double* matriz_host, int filas, int columnas, int rango_matriz, 
                                           char* tipo_resultado, double* tiempo_total,
                                           int bloques_cuda, int hilos_por_bloque) {
    
    if (!matriz_host || !tipo_resultado || !tiempo_total || 
        filas <= 0 || columnas <= 0 || rango_matriz <= 0 ||
        bloques_cuda <= 0 || hilos_por_bloque <= 0) {
        if (tiempo_total) *tiempo_total = 0.0;
        return NULL;
    }
    
    const double tiempo_inicio = obtener_tiempo_ms();
    
    if (rango_matriz == columnas && rango_matriz < filas) {
        // PSEUDOINVERSA IZQUIERDA: A+ = (A^T * A)^(-1) * A^T
        *tipo_resultado = 'L';
        
        const size_t tamano_A = filas * columnas * sizeof(double);
        const size_t tamano_At = columnas * filas * sizeof(double);     
        const size_t tamano_AtA = columnas * columnas * sizeof(double);
        
        double *gpu_A, *gpu_A_t, *gpu_AtA, *gpu_AtA_inv, *gpu_L;
        
        // Reservar memoria GPU
        if (cudaMalloc(&gpu_A, tamano_A) != cudaSuccess ||
            cudaMalloc(&gpu_A_t, tamano_At) != cudaSuccess ||
            cudaMalloc(&gpu_AtA, tamano_AtA) != cudaSuccess ||
            cudaMalloc(&gpu_AtA_inv, tamano_AtA) != cudaSuccess ||
            cudaMalloc(&gpu_L, tamano_At) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); 
            cudaFree(gpu_AtA_inv); cudaFree(gpu_L);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar datos y configurar kernels
        if (cudaMemcpy(gpu_A, matriz_host, tamano_A, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); 
            cudaFree(gpu_AtA_inv); cudaFree(gpu_L);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Configuración optimizada para kernels 2D
        const int threads_per_dim = (int)sqrt(hilos_por_bloque * hilos_por_bloque);
        const int optimal_threads = (threads_per_dim <= 32) ? threads_per_dim : 16;
        const dim3 block(optimal_threads, optimal_threads);
        const dim3 grid_t((columnas + block.x - 1) / block.x, (filas + block.y - 1) / block.y);
        const dim3 grid_m((columnas + block.x - 1) / block.x, (columnas + block.y - 1) / block.y);
        
        // Ejecutar kernels paralelos
        kernel_transponer<<<grid_t, block>>>(gpu_A, gpu_A_t, filas, columnas);
        cudaDeviceSynchronize();
        
        kernel_multiplicar<<<grid_m, block>>>(gpu_A_t, gpu_A, gpu_AtA, columnas, filas, columnas);
        cudaDeviceSynchronize();
        
        // Inversión LU paralela en GPU (ALGORITMO ÚNICO Y ÓPTIMO)
        double* host_AtA = (double*)malloc(tamano_AtA);
        cudaMemcpy(host_AtA, gpu_AtA, tamano_AtA, cudaMemcpyDeviceToHost);
        
        double* host_AtA_inv = invertir_matriz_lu_cuda(host_AtA, columnas);
        if (!host_AtA_inv) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); 
            cudaFree(gpu_AtA_inv); cudaFree(gpu_L);
            free(host_AtA);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar resultado de inversión a GPU
        cudaMemcpy(gpu_AtA_inv, host_AtA_inv, tamano_AtA, cudaMemcpyHostToDevice);
        
        // Multiplicación final paralela
        const dim3 grid_f((filas + block.x - 1) / block.x, (columnas + block.y - 1) / block.y);
        kernel_multiplicar<<<grid_f, block>>>(gpu_AtA_inv, gpu_A_t, gpu_L, columnas, columnas, filas);
        cudaDeviceSynchronize();
        
        // Copiar resultado final
        double* resultado = (double*)malloc(tamano_At);
        if (!resultado || cudaMemcpy(resultado, gpu_L, tamano_At, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); 
            cudaFree(gpu_AtA_inv); cudaFree(gpu_L);
            free(host_AtA); free(host_AtA_inv); free(resultado);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Limpiar memoria
        cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_AtA_inv); cudaFree(gpu_L);
        free(host_AtA); free(host_AtA_inv);
        
        *tiempo_total = obtener_tiempo_ms() - tiempo_inicio;
        return resultado;
        
    } else if (rango_matriz == filas && rango_matriz < columnas) {
        // PSEUDOINVERSA DERECHA: A+ = A^T * (A * A^T)^(-1)
        *tipo_resultado = 'R';
        
        const size_t tamano_A = filas * columnas * sizeof(double);
        const size_t tamano_At = columnas * filas * sizeof(double);     
        const size_t tamano_AAt = filas * filas * sizeof(double);
        
        double *gpu_A, *gpu_A_t, *gpu_AAt, *gpu_AAt_inv, *gpu_R;
        
        // Reservar memoria GPU
        if (cudaMalloc(&gpu_A, tamano_A) != cudaSuccess ||
            cudaMalloc(&gpu_A_t, tamano_At) != cudaSuccess ||
            cudaMalloc(&gpu_AAt, tamano_AAt) != cudaSuccess ||
            cudaMalloc(&gpu_AAt_inv, tamano_AAt) != cudaSuccess ||
            cudaMalloc(&gpu_R, tamano_At) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); 
            cudaFree(gpu_AAt_inv); cudaFree(gpu_R);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar datos y configurar kernels
        if (cudaMemcpy(gpu_A, matriz_host, tamano_A, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); 
            cudaFree(gpu_AAt_inv); cudaFree(gpu_R);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Configuración optimizada para kernels 2D
        const int threads_per_dim = (int)sqrt(hilos_por_bloque * hilos_por_bloque);
        const int optimal_threads = (threads_per_dim <= 32) ? threads_per_dim : 16;
        const dim3 block(optimal_threads, optimal_threads);
        const dim3 grid_t((columnas + block.x - 1) / block.x, (filas + block.y - 1) / block.y);
        const dim3 grid_m((filas + block.x - 1) / block.x, (filas + block.y - 1) / block.y);
        
        // Ejecutar kernels paralelos
        kernel_transponer<<<grid_t, block>>>(gpu_A, gpu_A_t, filas, columnas);
        cudaDeviceSynchronize();
        
        kernel_multiplicar<<<grid_m, block>>>(gpu_A, gpu_A_t, gpu_AAt, filas, columnas, filas);
        cudaDeviceSynchronize();
        
        // Inversión LU paralela en GPU (ALGORITMO ÚNICO Y ÓPTIMO)
        double* host_AAt = (double*)malloc(tamano_AAt);
        cudaMemcpy(host_AAt, gpu_AAt, tamano_AAt, cudaMemcpyDeviceToHost);
        
        double* host_AAt_inv = invertir_matriz_lu_cuda(host_AAt, filas);
        if (!host_AAt_inv) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); 
            cudaFree(gpu_AAt_inv); cudaFree(gpu_R);
            free(host_AAt);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar resultado de inversión a GPU
        cudaMemcpy(gpu_AAt_inv, host_AAt_inv, tamano_AAt, cudaMemcpyHostToDevice);
        
        // Multiplicación final paralela
        const dim3 grid_f((filas + block.x - 1) / block.x, (columnas + block.y - 1) / block.y);
        kernel_multiplicar<<<grid_f, block>>>(gpu_A_t, gpu_AAt_inv, gpu_R, columnas, filas, filas);
        cudaDeviceSynchronize();
        
        // Copiar resultado final
        double* resultado = (double*)malloc(tamano_At);
        if (!resultado || cudaMemcpy(resultado, gpu_R, tamano_At, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); 
            cudaFree(gpu_AAt_inv); cudaFree(gpu_R);
            free(host_AAt); free(host_AAt_inv); free(resultado);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Limpiar memoria
        cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_AAt_inv); cudaFree(gpu_R);
        free(host_AAt); free(host_AAt_inv);
        
        *tiempo_total = obtener_tiempo_ms() - tiempo_inicio;
        return resultado;
        
    } else {
        *tiempo_total = 0.0;
        return NULL;
    }
}


/**
 * FUNCIÓN PRINCIPAL DEL PROGRAMA
 * 
 * Archivos generados:
 * - salida.sal: contiene la pseudoinversa calculada
 * - metrica.met: contiene las métricas de optimización CUDA
 */
int main() {
    printf(" === PROGRAMA PSEUDOINVERSA CUDA OPTIMIZADO ===\n\n");
    
    // ========================================
    // PASO 1: LECTURA Y CARGA  DE LA MATRIZ
    // ========================================
    double* matriz_entrada = NULL;  // Matriz en memoria del host (CPU)
    int numero_filas, numero_columnas;
    
    printf("  Leyendo matriz de entrada...\n");
    leer_matriz("Entrada_matrices/entrada_1.ent", &matriz_entrada, &numero_filas, &numero_columnas);
    printf("  Matriz %dx%d cargada exitosamente\n", numero_filas, numero_columnas);
    
    // ========================================  
    // PASO 2: ANÁLISIS MATEMÁTICO
    // ========================================
    printf("\n  === ANÁLISIS MATEMÁTICO PARALELO ===\n");
    // Calcular rango usando algoritmo paralelo CUDA
    const int rango_calculado = calcular_rango_cuda(matriz_entrada, numero_filas, numero_columnas);
    printf(" Análisis completado con algoritmo paralelo:\n");
    printf("   - Rango calculado: %d\n", rango_calculado);
    printf("   - Dimensiones: %dx%d\n", numero_filas, numero_columnas);
    printf("   - Elementos totales: %d\n", numero_filas * numero_columnas);
    printf("   - Algoritmo: 100%% PARALELO CUDA\n");
    
    // Determinar qué tipo de pseudoinversa es posible calcular
    bool puede_calcular_pseudoinversa = false;
    char tipo_esperado = '?';
    
    if (rango_calculado == numero_filas && rango_calculado < numero_columnas) {
        printf(" PSEUDOINVERSA DERECHA (R): más columnas que filas, rango completo en filas\n");
        printf("   Formula: A^+ = A^T * (A * A^T)^(-1)\n");
        puede_calcular_pseudoinversa = true;
        tipo_esperado = 'R';
    } else if (rango_calculado == numero_columnas && rango_calculado < numero_filas) {
        printf(" PSEUDOINVERSA IZQUIERDA (L): más filas que columnas, rango completo en columnas\n");
        printf("   Formula: A^+ = (A^T * A)^(-1) * A^T\n");
        puede_calcular_pseudoinversa = true;
        tipo_esperado = 'L';
    } else if (rango_calculado == numero_filas && rango_calculado == numero_columnas) {
        printf(" MATRIZ CUADRADA INVERTIBLE: usar inversión estándar\n");
        printf("   Formula: A^+ = A^(-1)\n");
        puede_calcular_pseudoinversa = true;
        tipo_esperado = 'I'; // Invertible
    } else {
        printf(" SIN PSEUDOINVERSA: rango deficiente\n");
        printf("   Rango actual: %d, Requerido: %d (filas) o %d (columnas)\n", 
               rango_calculado, numero_filas, numero_columnas);
        puede_calcular_pseudoinversa = false;
    }
    
    if (!puede_calcular_pseudoinversa) {
        printf("\n No es posible calcular la pseudoinversa\n");
        guardar_sin_pseudoinversa();
        free(matriz_entrada);
        return 0;
    }

    // =========================================
    // PASO 3: CÁLCULO 100% PARALELO CUDA
    // =========================================
    printf("\n === CÁLCULO 100%% PARALELO CUDA ===\n");
    
    // Configuración óptima usando potencias de 2 para mejor rendimiento CUDA
    const int bloques_configuracion_optima = 32;
    const int hilos_configuracion_optima = 16;  // 16 hilos por dimensión (16x16 = 256 total)
    
    printf(" Configuración principal: %d bloques, %d hilos por dimensión\n", 
           bloques_configuracion_optima, hilos_configuracion_optima);
    
    char tipo_pseudoinversa_resultado;
    double tiempo_calculo_principal;
    double* pseudoinversa_calculada = calcular_pseudoinversa_cuda_paralela(matriz_entrada, numero_filas, numero_columnas, 
                                                                          rango_calculado, &tipo_pseudoinversa_resultado, 
                                                                          &tiempo_calculo_principal,
                                                                          bloques_configuracion_optima, hilos_configuracion_optima);
    
    if (!pseudoinversa_calculada) {
        printf(" Error en cálculo 100%% paralelo CUDA\n");
        guardar_sin_pseudoinversa();
        free(matriz_entrada);
        return 0;
    }
    
    printf(" Cálculo 100%% paralelo completado en %.6f ms\n", tiempo_calculo_principal);
    printf(" Tipo de pseudoinversa calculada: %c (esperado: %c)\n", 
           tipo_pseudoinversa_resultado, tipo_esperado);
    
    // Calcular dimensiones optimizadas de la pseudoinversa
    // Para pseudoinversa L: A+ tiene dimensiones n x m
    // Para pseudoinversa R: A+ tiene dimensiones n x m  
    const int pseudoinversa_filas = numero_columnas;    // Siempre n (columnas de A)
    const int pseudoinversa_columnas = numero_filas;    // Siempre m (filas de A)
    
    printf("📏 Dimensiones pseudoinversa: %dx%d\n", pseudoinversa_filas, pseudoinversa_columnas);
    
    guardar_pseudoinversa(pseudoinversa_calculada, pseudoinversa_filas, pseudoinversa_columnas, tipo_pseudoinversa_resultado);

    // ==========================================
    // PASO 4: ENSAYOS Y CÁLCULO DE SPEEDUP
    // ==========================================
    printf("\n === ENSAYOS Y CÁLCULO DE SPEEDUP ===\n");
    
    // Configuraciones optimizadas usando potencias de 2 para mejor eficiencia
    const int total_ensayos_benchmark = 12;
    // Configuraciones balanceadas: bloques x hilos = carga total equilibrada
    int configuraciones_bloques[] = {8, 16, 32, 64, 16, 32, 64, 128, 32, 64, 128, 256};
    int configuraciones_hilos[] = {8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32};
    
    double* tiempos_ensayos = (double*)malloc(total_ensayos_benchmark * sizeof(double));
    if (!tiempos_ensayos) {
        printf(" Error: No se pudo reservar memoria para tiempos de ensayos\n");
        free(matriz_entrada); free(pseudoinversa_calculada);
        return 1;
    }
    
    printf(" Ejecutando %d configuraciones para análisis de speedup:\n", total_ensayos_benchmark);
    
    // Ejecutar cada configuración y medir tiempos
    for (int indice_ensayo = 0; indice_ensayo < total_ensayos_benchmark; indice_ensayo++) {
        const int bloques_ensayo = configuraciones_bloques[indice_ensayo];
        const int hilos_ensayo = configuraciones_hilos[indice_ensayo];
        
        printf(" Ensayo %d/%d: %d bloques, %d hilos ", 
               indice_ensayo + 1, total_ensayos_benchmark, bloques_ensayo, hilos_ensayo);
        
        char tipo_temporal;
        double tiempo_temporal;
        double* resultado_temporal = calcular_pseudoinversa_cuda_paralela(matriz_entrada, numero_filas, numero_columnas, 
                                                                        rango_calculado, &tipo_temporal, &tiempo_temporal,
                                                                        bloques_ensayo, hilos_ensayo);
        
        if (resultado_temporal) {
            tiempos_ensayos[indice_ensayo] = tiempo_temporal;
            printf("-> %.6f ms\n", tiempo_temporal);
            free(resultado_temporal);
        } else {
            tiempos_ensayos[indice_ensayo] = 999999.0;
            printf("-> FALLÓ\n");
        }
    }
    
    // ==========================================
    // PASO 5: ANÁLISIS DE SPEEDUP Y RENDIMIENTO
    // ==========================================
    printf("\n === ANÁLISIS DE SPEEDUP Y RENDIMIENTO ===\n");
    
    double tiempo_mejor_ensayo = tiempos_ensayos[0];
    int indice_configuracion_optima = 0;
    double tiempo_peor_ensayo = tiempos_ensayos[0];
    double suma_tiempos = 0.0;
    int ensayos_exitosos = 0;
    
    // Análisis estadístico
    for (int i = 0; i < total_ensayos_benchmark; i++) {
        const double tiempo_actual = tiempos_ensayos[i];
        
        if (tiempo_actual < 999999.0) {
            ensayos_exitosos++;
            suma_tiempos += tiempo_actual;
            
            if (tiempo_actual < tiempo_mejor_ensayo) {
                tiempo_mejor_ensayo = tiempo_actual;
                indice_configuracion_optima = i;
            }
            if (tiempo_actual > tiempo_peor_ensayo) {
                tiempo_peor_ensayo = tiempo_actual;
            }
        }
    }
    
    // Calcular speedup usando tiempo secuencial definido
    const double tiempo_secuencial = TIEMPO_SECUENCIAL_MS;
    const double speedup = (tiempo_mejor_ensayo > 0) ? (tiempo_secuencial / tiempo_mejor_ensayo) : 0.0;
    const double tiempo_promedio = (ensayos_exitosos > 0) ? (suma_tiempos / ensayos_exitosos) : 0.0;
    const double mejora_relativa = (tiempo_peor_ensayo > 0) ? (tiempo_peor_ensayo / tiempo_mejor_ensayo) : 1.0;
    
    printf("\n === RESULTADOS DE SPEEDUP ===\n");
    printf(" Tiempo secuencial (referencia): %.6f ms\n", tiempo_secuencial);
    printf(" Tiempo paralelo (mejor): %.6f ms\n", tiempo_mejor_ensayo);
    printf(" SPEEDUP = %.2fx\n", speedup);
    printf(" Mejor configuración: %d bloques, %d hilos\n", 
           configuraciones_bloques[indice_configuracion_optima], 
           configuraciones_hilos[indice_configuracion_optima]);
    printf(" Tiempo promedio: %.6f ms\n", tiempo_promedio);
    printf(" Mejora relativa: %.2fx (mejor vs peor)\n", mejora_relativa);
    printf(" Ensayos exitosos: %d/%d\n", ensayos_exitosos, total_ensayos_benchmark);
    
    // Evaluar eficiencia del speedup
    if (speedup > 1.0) {
        printf(" RESULTADO: Algoritmo paralelo es %.2fx más rápido que secuencial\n", speedup);
    } else if (speedup > 0.5) {
        printf("  RESULTADO: Algoritmo paralelo es competitivo (%.2fx)\n", speedup);
    } else {
        printf(" RESULTADO: Algoritmo paralelo es más lento que secuencial\n");
    }
    
    // Guardar métricas con speedup
    guardar_metricas_speedup(tiempo_secuencial, tiempo_mejor_ensayo, tiempos_ensayos, 
                            configuraciones_bloques, configuraciones_hilos, total_ensayos_benchmark);

    // ==========================================
    // PASO 6: FINALIZACIÓN Y RESUMEN
    // ==========================================
    printf("\n === PROGRAMA 100%% PARALELO COMPLETADO ===\n");
    printf(" Archivos generados:\n");
    printf("   - salida.sal (pseudoinversa %dx%d, tipo %c)\n", 
           pseudoinversa_filas, pseudoinversa_columnas, tipo_pseudoinversa_resultado);
    printf("   - metrica.met (speedup %.2fx y %d configuraciones)\n", speedup, total_ensayos_benchmark);
    printf(" Algoritmo: 100%% PARALELO CUDA SIN SECUENCIALES\n");
    printf(" Mejor rendimiento: %.6f ms (speedup %.2fx)\n", tiempo_mejor_ensayo, speedup);
    printf(" Tiempo secuencial referencia: %.6f ms\n", tiempo_secuencial);
    
    // Nota importante sobre tiempo secuencial
    printf("\n NOTA: Para actualizar el tiempo secuencial de referencia:\n");
    printf("   1. Modifica la constante TIEMPO_SECUENCIAL_MS en línea %d\n", __LINE__ - 30);
    printf("   2. Recompila el programa con tu tiempo secuencial medido\n");
    printf("   3. El speedup se calculará automáticamente\n");
    
    // Liberar toda la memoria dinámica de forma segura
    free(matriz_entrada);
    free(pseudoinversa_calculada);
    free(tiempos_ensayos);
    
    printf(" Programa terminado exitosamente\n");
    return 0;
}
