/* Autores: Francisco Soto Lagos, Sebastian Salinas Jorquera
 * Implementación completamente paralela para cálculo de pseudoinversa de matrices
 * 
 * FUNCIONALIDAD:
 * 1. Lee una matriz desde archivo "entrada.ent" 
 * 2. Calcula el rango usando algoritmos paralelos CUDA
 * 3. Determina el tipo de pseudoinversa (izquierda o derecha)
 * 4. Calcula la pseudoinversa usando algoritmos CUDA optimizados
 * 5. Guarda el resultado en "salida.sal"
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

// Constantes y configuraciones
#define EPSILON 1e-12
#define MAX_PRECISION 15
#define NUM_ENSAYOS 10
#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024



// Estructura para manejar múltiples punteros GPU
typedef struct {
    void** punteros;
    size_t* tamanos;
    int count;
} MemoriaGPU;

// Función utilitaria para reserva masiva de memoria GPU
bool reservar_memoria_gpu(MemoriaGPU* mem) {
    bool exito = true;
    
    // Intentar reservar toda la memoria
    for (int i = 0; i < mem->count; i++) {
        if (cudaMalloc((void**)mem->punteros[i], mem->tamanos[i]) != cudaSuccess) {
            exito = false;
            break;
        }
    }
    
    // Si falló alguna reserva, liberar todo lo reservado hasta ahora
    if (!exito) {
        for (int i = 0; i < mem->count; i++) {
            if (mem->punteros[i] && *(void**)mem->punteros[i]) {
                cudaFree(*(void**)mem->punteros[i]);
                *(void**)mem->punteros[i] = NULL;
            }
        }
    }
    
    return exito;
}

// Función utilitaria para liberar memoria GPU
void liberar_memoria_gpu(MemoriaGPU* mem) {
    for (int i = 0; i < mem->count; i++) {
        if (mem->punteros[i] && *(void**)mem->punteros[i]) {
            cudaFree(*(void**)mem->punteros[i]);
            *(void**)mem->punteros[i] = NULL;
        }
    }
}

// Funciones utilitarias
double obtener_tiempo_ms() {
    #ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart * 1000.0;
    #else
    // Para sistemas no Windows, usar clock()
    return (double)clock() / CLOCKS_PER_SEC * 1000.0;
    #endif
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
 * Función para guardar la pseudoinversa en archivo de salida
 * 
 * Formato del archivo salida.sal:
 * Ltipo de pseudoinversa ('L' o 'R')
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

// Función para guardar métricas de múltiples ensayos
void guardar_metricas_multiples_ensayos(int ensayos[][3], double tiempos[], int num_ensayos) {
    FILE* archivo_metricas = fopen("metrica.met", "w");
    if (!archivo_metricas) {
        printf(" ERROR: No se pudo crear el archivo metrica.met\n");
        return;
    }
    
    // Escribir encabezados de la tabla
    fprintf(archivo_metricas, "%-8s %-12s %-18s %-12s\n", 
            "Ensayo", "Bloques", "Hilos_por_Bloque", "Speedup");
    fprintf(archivo_metricas, "%-8s %-12s %-18s %-12s\n", 
            "------", "-------", "----------------", "-------");
    
    // Escribir datos de todos los ensayos realizados
    for (int i = 0; i < num_ensayos; i++) {
        fprintf(archivo_metricas, "%-8d %-12d %-18d %-12.6f\n", 
                i + 1, ensayos[i][0], ensayos[i][1], 1.0);
    }
    
    fclose(archivo_metricas);
    printf("  Métricas de %d ensayos guardadas en metrica.met\n", num_ensayos);
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
    
    // Reservar memoria GPU usando función utilitaria
    void* punteros[] = {(void**)&gpu_matriz, (void**)&gpu_max_values, (void**)&gpu_max_indices};
    size_t tamanos[] = {size, filas * sizeof(double), filas * sizeof(int)};
    MemoriaGPU mem = {punteros, tamanos, 3};
    
    if (!reservar_memoria_gpu(&mem)) {
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
    liberar_memoria_gpu(&mem);
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
    
    // Reservar memoria GPU usando función utilitaria
    void* punteros[] = {(void**)&gpu_matriz, (void**)&gpu_identidad, (void**)&gpu_resultado, 
                        (void**)&gpu_temp_y, (void**)&gpu_permutaciones, (void**)&gpu_pivot_row, 
                        (void**)&gpu_pivot_value};
    size_t tamanos[] = {size, size, size, size, n * sizeof(int), sizeof(int), sizeof(double)};
    MemoriaGPU mem = {punteros, tamanos, 7};
    
    if (!reservar_memoria_gpu(&mem)) {
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
    liberar_memoria_gpu(&mem);
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
 * KERNEL CUDA, Multiplicar matrices en paralelo, el kernel_multiplicar() implementa el algoritmo clasico de
 multiplicacion de matrices, paralelizado por elementos. Cada hilo CUDA calcula una celda de la matriz resultado.
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
 * 
 * Parámetros:
 *   - bloques_cuda: Se valida pero no se usa directamente (kernels 2D calculan grid automáticamente)
 *   - hilos_por_bloque: Usado para configurar dim3 block(hilos_por_bloque, hilos_por_bloque)
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
        
        // Reservar memoria GPU usando función utilitaria
        void* punteros[] = {(void**)&gpu_A, (void**)&gpu_A_t, (void**)&gpu_AtA, 
                            (void**)&gpu_AtA_inv, (void**)&gpu_L};
        size_t tamanos[] = {tamano_A, tamano_At, tamano_AtA, tamano_AtA, tamano_At};
        MemoriaGPU mem = {punteros, tamanos, 5};
        
        if (!reservar_memoria_gpu(&mem)) {
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar datos y configurar kernels
        if (cudaMemcpy(gpu_A, matriz_host, tamano_A, cudaMemcpyHostToDevice) != cudaSuccess) {
            liberar_memoria_gpu(&mem);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Configuración de kernels usando parámetros especificados
        // Nota: Para kernels 2D, el número de bloques se calcula automáticamente
        // basado en las dimensiones de la matriz y hilos_por_bloque
        const dim3 block(hilos_por_bloque, hilos_por_bloque);
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
            liberar_memoria_gpu(&mem);
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
            liberar_memoria_gpu(&mem);
            free(host_AtA); free(host_AtA_inv); free(resultado);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Limpiar memoria
        liberar_memoria_gpu(&mem);
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
        
        // Reservar memoria GPU usando función utilitaria
        void* punteros[] = {(void**)&gpu_A, (void**)&gpu_A_t, (void**)&gpu_AAt, 
                            (void**)&gpu_AAt_inv, (void**)&gpu_R};
        size_t tamanos[] = {tamano_A, tamano_At, tamano_AAt, tamano_AAt, tamano_At};
        MemoriaGPU mem = {punteros, tamanos, 5};
        
        if (!reservar_memoria_gpu(&mem)) {
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar datos y configurar kernels
        if (cudaMemcpy(gpu_A, matriz_host, tamano_A, cudaMemcpyHostToDevice) != cudaSuccess) {
            liberar_memoria_gpu(&mem);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Configuración de kernels usando parámetros especificados
        // Nota: Para kernels 2D, el número de bloques se calcula automáticamente
        // basado en las dimensiones de la matriz y hilos_por_bloque
        const dim3 block(hilos_por_bloque, hilos_por_bloque);
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
            liberar_memoria_gpu(&mem);
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
            liberar_memoria_gpu(&mem);
            free(host_AAt); free(host_AAt_inv); free(resultado);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Limpiar memoria
        liberar_memoria_gpu(&mem);
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
 * Archivo generado:
 * - salida.sal: contiene la pseudoinversa calculada
 */
int main() {
    printf(" === PROGRAMA CÁLCULO PSEUDOINVERSA CUDA ===\n\n");
    
    // ========================================
    // PASO 1: LECTURA Y CARGA  DE LA MATRIZ
    // ========================================
    double* matriz_entrada = NULL;  // Matriz en memoria del host (CPU)
    int numero_filas, numero_columnas;
    
    printf("  Leyendo matriz de entrada...\n");
    leer_matriz("entrada.ent", &matriz_entrada, &numero_filas, &numero_columnas);
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
    // PASO 3: MÚLTIPLES ENSAYOS PARALELOS CUDA
    // =========================================
    printf("\n === MÚLTIPLES ENSAYOS PARALELOS CUDA ===\n");
    
    // Configuraciones para 10 ensayos diferentes
    int configuraciones[10][2] = {
        {8, 8},     // Ensayo 1: 8 bloques, 8 hilos
        {16, 8},    // Ensayo 2: 16 bloques, 8 hilos  
        {16, 16},   // Ensayo 3: 16 bloques, 16 hilos
        {32, 8},    // Ensayo 4: 32 bloques, 8 hilos
        {32, 16},   // Ensayo 5: 32 bloques, 16 hilos
        {64, 8},    // Ensayo 6: 64 bloques, 8 hilos
        {64, 16},   // Ensayo 7: 64 bloques, 16 hilos
        {128, 8},   // Ensayo 8: 128 bloques, 8 hilos
        {128, 16},  // Ensayo 9: 128 bloques, 16 hilos
        {256, 16}   // Ensayo 10: 256 bloques, 16 hilos
    };
    
    int ensayos_realizados[10][3]; // [bloques, hilos, resultado]
    double tiempos_ensayos[10];
    double* pseudoinversa_final = NULL;
    char tipo_pseudoinversa_resultado;
    int ensayos_exitosos = 0;
    
    for (int ensayo = 0; ensayo < 10; ensayo++) {
        int bloques = configuraciones[ensayo][0];
        int hilos = configuraciones[ensayo][1];
        
        printf(" Ensayo %d: %d bloques, %d hilos por dimensión... ", 
               ensayo + 1, bloques, hilos);
        
        char tipo_temp;
        double tiempo_temp;
        double* resultado_temp = calcular_pseudoinversa_cuda_paralela(
            matriz_entrada, numero_filas, numero_columnas, 
            rango_calculado, &tipo_temp, &tiempo_temp,
            bloques, hilos);
        
        if (resultado_temp) {
            printf("✓ %.3f ms\n", tiempo_temp);
            ensayos_realizados[ensayos_exitosos][0] = bloques;
            ensayos_realizados[ensayos_exitosos][1] = hilos;
            ensayos_realizados[ensayos_exitosos][2] = 1; // exitoso
            tiempos_ensayos[ensayos_exitosos] = tiempo_temp;
            
            // Guardar el primer resultado exitoso para salida
            if (ensayos_exitosos == 0) {
                pseudoinversa_final = resultado_temp;
                tipo_pseudoinversa_resultado = tipo_temp;
            } else {
                free(resultado_temp);
            }
            ensayos_exitosos++;
        } else {
            printf("✗ Error\n");
        }
    }
    
    if (ensayos_exitosos == 0) {
        printf(" Error: Ningún ensayo fue exitoso\n");
        guardar_sin_pseudoinversa();
        free(matriz_entrada);
        return 0;
    }
    
    printf(" Completados %d/10 ensayos exitosos\n", ensayos_exitosos);
    printf(" Tipo de pseudoinversa calculada: %c (esperado: %c)\n", 
           tipo_pseudoinversa_resultado, tipo_esperado);
    
    // Calcular dimensiones optimizadas de la pseudoinversa
    const int pseudoinversa_filas = numero_columnas;    // Siempre n (columnas de A)
    const int pseudoinversa_columnas = numero_filas;    // Siempre m (filas de A)
    
    printf("📏 Dimensiones pseudoinversa: %dx%d\n", pseudoinversa_filas, pseudoinversa_columnas);
    
    guardar_pseudoinversa(pseudoinversa_final, pseudoinversa_filas, pseudoinversa_columnas, tipo_pseudoinversa_resultado);
    guardar_metricas_multiples_ensayos(ensayos_realizados, tiempos_ensayos, ensayos_exitosos);

    // ==========================================
    // PASO 4: FINALIZACIÓN DEL PROGRAMA
    // ==========================================
    printf("\n === PROGRAMA COMPLETADO EXITOSAMENTE ===\n");
    printf(" Archivos generados:\n");
    printf("   - salida.sal (pseudoinversa %dx%d, tipo %c)\n", 
           pseudoinversa_filas, pseudoinversa_columnas, tipo_pseudoinversa_resultado);
    printf("   - metrica.met (%d ensayos realizados)\n", ensayos_exitosos);
    printf(" Algoritmo: 100%% PARALELO CUDA\n");
    
    // Liberar toda la memoria dinámica de forma segura
    free(matriz_entrada);
    free(pseudoinversa_final);
    
    printf(" Programa terminado exitosamente\n");
    return 0;
}
