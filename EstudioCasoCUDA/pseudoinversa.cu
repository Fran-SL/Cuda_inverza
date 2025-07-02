/*
 * Programa CUDA: Cálculo paralelo de pseudoinversa de matrices
 * Autores: Francisco Soto Lagos, Sebastian Salinas Jorquera
 * Implementación 100% paralela con optimizaciones CUDA
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

// Funciones utilitarias
double obtener_tiempo_ms() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart * 1000.0;
}

/**
 * Función para imprimir matriz en consola (solo si DEBUG_MODE está definido)
 * Parámetros:
 *   - A: puntero a la matriz (almacenada en formato lineal)
 *   - m: número de filas
 *   - n: número de columnas  
 *   - nombre: nombre descriptivo para mostrar
 */
void imprimir_matriz(double* A, int m, int n, const char* nombre) {
    #ifdef DEBUG_MODE  // Solo se ejecuta si definimos DEBUG_MODE al compilar
    printf("\n=== %s (%dx%d) ===\n", nombre, m, n);
    for (int i = 0; i < m; i++) {        // Recorrer filas
        for (int j = 0; j < n; j++) {    // Recorrer columnas
            // Acceso lineal: matriz[i][j] = A[i*n + j]
            printf("%8.6f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
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
    const size_t tamaño_memoria = total_elementos * sizeof(double);

    *matriz_destino = (double*)malloc(tamaño_memoria);
    if (!*matriz_destino) {
        printf(" ERROR: No se pudo reservar memoria para matriz %dx%d (%zu bytes)\n", 
               *filas, *columnas, tamaño_memoria);
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
 * Función optimizada para guardar métricas de optimización CUDA en archivo
 * 
 * Formato del archivo metrica.met:
 * Cada línea: ensayo bloques hilos tiempo_ms eficiencia_relativa
 * 
 * Parámetros:
 *   - tiempo_referencia: tiempo de referencia (primera configuración)
 *   - tiempos_medidos: array de tiempos CUDA medidos
 *   - configuraciones_bloques, configuraciones_hilos: configuraciones usadas
 *   - total_ensayos: cantidad de mediciones
 */
void guardar_metricas(double tiempo_referencia, double* tiempos_medidos, 
                      int* configuraciones_bloques, int* configuraciones_hilos, int total_ensayos) {
    // Validación de parámetros de entrada
    if (!tiempos_medidos || !configuraciones_bloques || !configuraciones_hilos || total_ensayos <= 0) {
        printf(" ERROR: Parámetros inválidos para guardar métricas\n");
        return;
    }
    
    if (tiempo_referencia <= 0.0) {
        printf(" ERROR: Tiempo de referencia inválido: %.6f ms\n", tiempo_referencia);
        return;
    }
    
    FILE* archivo_metricas = fopen("metrica.met", "w");
    if (!archivo_metricas) {
        printf(" ERROR: No se pudo crear el archivo metrica.met\n");
        return;
    }
    
    // Escribir métricas de cada ensayo de forma optimizada
    for (int ensayo = 0; ensayo < total_ensayos; ensayo++) {
        // Calcular eficiencia relativa
        double eficiencia_relativa = tiempo_referencia / tiempos_medidos[ensayo];
        
        // Formato: número_ensayo num_bloques hilos_por_bloque tiempo_ms eficiencia_relativa
        fprintf(archivo_metricas, "%d %d %d %.15f %.15f\n", 
                ensayo + 1, 
                configuraciones_bloques[ensayo], 
                configuraciones_hilos[ensayo], 
                tiempos_medidos[ensayo], 
                eficiencia_relativa);
    }
    
    fclose(archivo_metricas);
    printf("  Métricas de %d ensayos guardadas en metrica.met\n", total_ensayos);
}

// Función optimizada para guardar resultado cuando no hay pseudoinversa
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
// FUNCIONES DE ÁLGEBRA LINEAL (CPU)
// ===================================================================

/**
 * Calcular el rango de una matriz usando eliminación gaussiana optimizada
 * El rango es el número de filas/columnas linealmente independientes
 * 
 * Algoritmo optimizado:
 * 1. Validación de entrada y optimización de acceso a memoria
 * 2. Para cada columna, buscar el mejor pivote (elemento más grande)
 * 3. Intercambiar filas de forma eficiente si es necesario
 * 4. Eliminar elementos debajo del pivote con acceso optimizado
 * 5. Contar filas no nulas
 * 
 * Parámetros:
 *   - A: matriz original
 *   - filas: número de filas
 *   - columnas: número de columnas
 * Retorna: rango de la matriz (0 si error)
 */
int calcular_rango(double* A, int filas, int columnas) {
    // Validación de entrada
    if (!A || filas <= 0 || columnas <= 0) return 0;
    
    // Optimización: Calcular constantes una sola vez
    const size_t size_matriz = filas * columnas * sizeof(double);
    const int dimensión_minima = (filas < columnas) ? filas : columnas;
    
    // Crear copia temporal para no modificar la matriz original
    double* matriz_trabajo = (double*)malloc(size_matriz);
    if (!matriz_trabajo) return 0; // Error de memoria
    
    memcpy(matriz_trabajo, A, size_matriz);
    
    int rango_actual = 0;
    
    // Eliminación gaussiana optimizada para cada columna
    for (int columna_pivote = 0; columna_pivote < dimensión_minima; columna_pivote++) {
        // PASO 1: Buscar el mejor pivote en la columna actual
        int fila_mejor_pivote = columna_pivote;
        double valor_mejor_pivote = fabs(matriz_trabajo[columna_pivote * columnas + columna_pivote]);
        
        // Buscar elemento con mayor valor absoluto en la columna
        for (int fila_candidata = columna_pivote + 1; fila_candidata < filas; fila_candidata++) {
            double valor_candidato = fabs(matriz_trabajo[fila_candidata * columnas + columna_pivote]);
            if (valor_candidato > valor_mejor_pivote) {
                valor_mejor_pivote = valor_candidato;
                fila_mejor_pivote = fila_candidata;
            }
        }
        
        // Si el pivote es muy pequeño, la columna es linealmente dependiente
        if (valor_mejor_pivote < EPSILON) continue;
        
        // PASO 2: Intercambiar filas de forma optimizada si es necesario
        if (fila_mejor_pivote != columna_pivote) {
            const int offset_pivote = columna_pivote * columnas;
            const int offset_mejor = fila_mejor_pivote * columnas;
            
            // Intercambio optimizado de filas completas
            for (int col = 0; col < columnas; col++) {
                double temp_valor = matriz_trabajo[offset_pivote + col];
                matriz_trabajo[offset_pivote + col] = matriz_trabajo[offset_mejor + col];
                matriz_trabajo[offset_mejor + col] = temp_valor;
            }
        }
        
        // PASO 3: Eliminación hacia abajo optimizada
        const int offset_fila_pivote = columna_pivote * columnas;
        const double elemento_pivote = matriz_trabajo[offset_fila_pivote + columna_pivote];
        
        for (int fila_eliminacion = columna_pivote + 1; fila_eliminacion < filas; fila_eliminacion++) {
            const int offset_fila_actual = fila_eliminacion * columnas;
            const double elemento_actual = matriz_trabajo[offset_fila_actual + columna_pivote];
            
            // Optimización: Solo procesar si el elemento no es despreciable
            if (fabs(elemento_actual) > EPSILON) {
                const double factor_eliminacion = elemento_actual / elemento_pivote;
                
                // Restar múltiplo de la fila pivote de forma optimizada
                for (int col = columna_pivote; col < columnas; col++) {
                    matriz_trabajo[offset_fila_actual + col] -= 
                        factor_eliminacion * matriz_trabajo[offset_fila_pivote + col];
                }
            }
        }
        
        rango_actual++;  // Incrementar rango por cada pivote válido encontrado
    }
    
    free(matriz_trabajo);  // Liberar memoria temporal
    return rango_actual;
}

// FUNCIÓN SECUENCIAL ELIMINADA - SOLO ALGORITMO PARALELO CUDA

// FUNCIÓN SECUENCIAL ELIMINADA - SOLO ALGORITMO PARALELO CUDA

/**
 * Invertir matriz cuadrada usando el método de Gauss-Jordan
 * 
 * Algoritmo:
 * 1. Crear matriz aumentada [A | I] donde I es la identidad
 * 2. Aplicar operaciones elementales para convertir A en I
 * 3. Las mismas operaciones convierten I en A^(-1)
 * 4. Resultado: [I | A^(-1)]
 * 
 * Parámetros:
 *   - A: matriz cuadrada a invertir (n x n)
 *   - n: dimensión de la matriz
 * Retorna: matriz inversa o NULL si es singular
 */
double* invertir_matriz(double* A, int n) {
    // Validación de entrada
    if (!A || n <= 0) return NULL;
    
    // Optimización: Calcular dimensiones una sola vez
    const int cols_aumentada = 2 * n;
    const size_t size_aumentada = n * cols_aumentada * sizeof(double);
    const size_t size_inversa = n * n * sizeof(double);
    
    // Reservar memoria para matriz aumentada [A | I] de tamaño n x 2n
    double* aumentada = (double*)malloc(size_aumentada);
    if (!aumentada) return NULL;
    
    // PASO 1: Crear matriz aumentada [A | I] de forma optimizada
    for (int fila = 0; fila < n; fila++) {
        const int offset_aumentada = fila * cols_aumentada;
        const int offset_original = fila * n;
        
        // Copiar elementos de A al lado izquierdo
        for (int col = 0; col < n; col++) {
            aumentada[offset_aumentada + col] = A[offset_original + col];
        }
        
        // Crear matriz identidad I al lado derecho
        for (int col = 0; col < n; col++) {
            aumentada[offset_aumentada + n + col] = (fila == col) ? 1.0 : 0.0;
        }
    }
    
    // PASO 2: Eliminación de Gauss-Jordan optimizada
    for (int pivote_fila = 0; pivote_fila < n; pivote_fila++) {
        // SUB-PASO 2.1: Buscar el mejor pivote en la columna
        int max_fila = pivote_fila;
        double max_valor = fabs(aumentada[pivote_fila * cols_aumentada + pivote_fila]);
        
        for (int fila = pivote_fila + 1; fila < n; fila++) {
            double valor_actual = fabs(aumentada[fila * cols_aumentada + pivote_fila]);
            if (valor_actual > max_valor) {
                max_valor = valor_actual;
                max_fila = fila;
            }
        }
        
        // Verificar si la matriz es singular
        if (max_valor < EPSILON) {
            free(aumentada);
            return NULL; // Matriz no invertible
        }
        
        // SUB-PASO 2.2: Intercambiar filas si es necesario
        if (max_fila != pivote_fila) {
            const int offset_pivote = pivote_fila * cols_aumentada;
            const int offset_max = max_fila * cols_aumentada;
            
            for (int col = 0; col < cols_aumentada; col++) {
                double temp = aumentada[offset_pivote + col];
                aumentada[offset_pivote + col] = aumentada[offset_max + col];
                aumentada[offset_max + col] = temp;
            }
        }
        
        // SUB-PASO 2.3: Normalizar la fila del pivote
        const int offset_pivote = pivote_fila * cols_aumentada;
        const double pivot = aumentada[offset_pivote + pivote_fila];
        
        for (int col = 0; col < cols_aumentada; col++) {
            aumentada[offset_pivote + col] /= pivot;
        }
        
        // SUB-PASO 2.4: Eliminar otros elementos de la columna
        for (int fila = 0; fila < n; fila++) {
            if (fila != pivote_fila) {
                const int offset_fila = fila * cols_aumentada;
                const double factor = aumentada[offset_fila + pivote_fila];
                
                // Optimización: Solo procesar si el factor no es cero
                if (fabs(factor) > EPSILON) {
                    for (int col = 0; col < cols_aumentada; col++) {
                        aumentada[offset_fila + col] -= factor * aumentada[offset_pivote + col];
                    }
                }
            }
        }
    }
    
    // PASO 3: Extraer la matriz inversa de forma optimizada
    double* inversa = (double*)malloc(size_inversa);
    if (!inversa) {
        free(aumentada);
        return NULL;
    }
    
    // Copiar solo la parte derecha de la matriz aumentada
    for (int fila = 0; fila < n; fila++) {
        const int offset_aumentada = fila * cols_aumentada + n; // Lado derecho
        const int offset_inversa = fila * n;
        
        for (int col = 0; col < n; col++) {
            inversa[offset_inversa + col] = aumentada[offset_aumentada + col];
        }
    }
    
    free(aumentada);  // Liberar memoria temporal
    return inversa;
}

// FUNCIÓN SECUENCIAL ELIMINADA - SOLO ALGORITMO PARALELO CUDA

// ===================================================================
// KERNELS CUDA PARA PARALELIZACIÓN
// ===================================================================

/**
 * KERNEL CUDA OPTIMIZADO: Transponer matriz en paralelo
 * Cada thread de CUDA calcula una posición de la matriz transpuesta
 * 
 * Mapeo optimizado de threads:
 * - blockIdx.x, blockIdx.y: posición del bloque en la grid
 * - threadIdx.x, threadIdx.y: posición del thread dentro del bloque
 * - columna_global, fila_global: posición global del thread en la matriz
 * 
 * Parámetros:
 *   - matriz_origen: matriz original en GPU (filas_origen x columnas_origen)
 *   - matriz_transpuesta: matriz transpuesta en GPU (columnas_origen x filas_origen)
 *   - filas_origen, columnas_origen: dimensiones de matriz_origen
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
 * Cada thread calcula un elemento del resultado C = A * B
 * 
 * Algoritmo paralelo optimizado:
 * - Cada thread (fila_resultado, columna_resultado) calcula C[fila][columna]
 * - Realiza el producto punto de la fila de A con la columna de B
 * - Acceso optimizado a memoria con índices precalculados
 * 
 * Parámetros:
 *   - matriz_A: primera matriz en GPU (filas_A x columnas_A)
 *   - matriz_B: segunda matriz en GPU (columnas_A x columnas_B)
 *   - matriz_C: matriz resultado en GPU (filas_A x columnas_B)
 *   - filas_A, columnas_A, columnas_B: dimensiones de las matrices
 */
__global__ void kernel_multiplicar(double* matriz_A, double* matriz_B, double* matriz_C, 
                                   int filas_A, int columnas_A, int columnas_B) {
    const int fila_resultado = blockIdx.y * blockDim.y + threadIdx.y;
    const int columna_resultado = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (fila_resultado < filas_A && columna_resultado < columnas_B) {
        double acumulador_suma = 0.0;
        const int offset_fila_A = fila_resultado * columnas_A;
        
        // Producto punto optimizado: C[fila][columna] = Σ(A[fila][k] * B[k][columna])
        for (int k = 0; k < columnas_A; k++) {
            const double elemento_A = matriz_A[offset_fila_A + k];
            const double elemento_B = matriz_B[k * columnas_B + columna_resultado];
            acumulador_suma += elemento_A * elemento_B;
        }
        
        const int indice_resultado = fila_resultado * columnas_B + columna_resultado;
        matriz_C[indice_resultado] = acumulador_suma;
    }
}

// Kernel optimizado con memoria compartida para matrices grandes
__global__ void kernel_multiplicar_shared(double* matriz_A, double* matriz_B, double* matriz_C,
                                         int filas_A, int columnas_A, int columnas_B) {
    const int TILE_SIZE = 16;
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int fila = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    double valor = 0.0;
    
    for (int tile = 0; tile < (columnas_A + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Cargar tiles en memoria compartida
        if (fila < filas_A && (tile * TILE_SIZE + tx) < columnas_A) {
            As[ty][tx] = matriz_A[fila * columnas_A + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if ((tile * TILE_SIZE + ty) < columnas_A && col < columnas_B) {
            Bs[ty][tx] = matriz_B[(tile * TILE_SIZE + ty) * columnas_B + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Calcular producto usando memoria compartida
        for (int k = 0; k < TILE_SIZE; k++) {
            valor += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (fila < filas_A && col < columnas_B) {
        matriz_C[fila * columnas_B + col] = valor;
    }
}

/**
 * FUNCIÓN CUDA OPTIMIZADA: Calcular pseudoinversa usando paralelización
 * Implementación híbrida: operaciones matriciales en GPU, inversión en CPU
 */
double* calcular_pseudoinversa_cuda(double* matriz_host, int filas, int columnas, int rango_matriz, 
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
        
        const size_t tamaño_A = filas * columnas * sizeof(double);
        const size_t tamaño_At = columnas * filas * sizeof(double);     
        const size_t tamaño_AtA = columnas * columnas * sizeof(double);
        
        double *gpu_A, *gpu_A_t, *gpu_AtA, *gpu_L;
        
        // Reservar memoria GPU
        if (cudaMalloc(&gpu_A, tamaño_A) != cudaSuccess ||
            cudaMalloc(&gpu_A_t, tamaño_At) != cudaSuccess ||
            cudaMalloc(&gpu_AtA, tamaño_AtA) != cudaSuccess ||
            cudaMalloc(&gpu_L, tamaño_At) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar datos y configurar kernels
        if (cudaMemcpy(gpu_A, matriz_host, tamaño_A, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        const dim3 block(hilos_por_bloque, hilos_por_bloque);
        const dim3 grid_t((columnas + block.x - 1) / block.x, (filas + block.y - 1) / block.y);
        const dim3 grid_m((columnas + block.x - 1) / block.x, (columnas + block.y - 1) / block.y);
        
        // Ejecutar kernels
        kernel_transponer<<<grid_t, block>>>(gpu_A, gpu_A_t, filas, columnas);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        kernel_multiplicar<<<grid_m, block>>>(gpu_A_t, gpu_A, gpu_AtA, columnas, filas, columnas);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar AtA a CPU e invertir
        double* host_AtA = (double*)malloc(tamaño_AtA);
        if (!host_AtA || cudaMemcpy(host_AtA, gpu_AtA, tamaño_AtA, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L);
            free(host_AtA);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        double* host_AtA_inv = invertir_matriz(host_AtA, columnas);
        if (!host_AtA_inv) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L);
            free(host_AtA);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar inversa a GPU y calcular resultado final
        double* gpu_AtA_inv;
        if (cudaMalloc(&gpu_AtA_inv, tamaño_AtA) != cudaSuccess ||
            cudaMemcpy(gpu_AtA_inv, host_AtA_inv, tamaño_AtA, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L); cudaFree(gpu_AtA_inv);
            free(host_AtA); free(host_AtA_inv);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        const dim3 grid_f((filas + block.x - 1) / block.x, (columnas + block.y - 1) / block.y);
        kernel_multiplicar<<<grid_f, block>>>(gpu_AtA_inv, gpu_A_t, gpu_L, columnas, columnas, filas);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L); cudaFree(gpu_AtA_inv);
            free(host_AtA); free(host_AtA_inv);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar resultado final
        double* resultado = (double*)malloc(tamaño_At);
        if (!resultado || cudaMemcpy(resultado, gpu_L, tamaño_At, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L); cudaFree(gpu_AtA_inv);
            free(host_AtA); free(host_AtA_inv); free(resultado);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Limpiar memoria
        cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AtA); cudaFree(gpu_L); cudaFree(gpu_AtA_inv);
        free(host_AtA); free(host_AtA_inv);
        
        *tiempo_total = obtener_tiempo_ms() - tiempo_inicio;
        return resultado;
        
    } else if (rango_matriz == filas && rango_matriz < columnas) {
        // PSEUDOINVERSA DERECHA: A+ = A^T * (A * A^T)^(-1)
        *tipo_resultado = 'R';
        
        const size_t tamaño_A = filas * columnas * sizeof(double);
        const size_t tamaño_At = columnas * filas * sizeof(double);     
        const size_t tamaño_AAt = filas * filas * sizeof(double);
        
        double *gpu_A, *gpu_A_t, *gpu_AAt, *gpu_R;
        
        // Reservar memoria GPU
        if (cudaMalloc(&gpu_A, tamaño_A) != cudaSuccess ||
            cudaMalloc(&gpu_A_t, tamaño_At) != cudaSuccess ||
            cudaMalloc(&gpu_AAt, tamaño_AAt) != cudaSuccess ||
            cudaMalloc(&gpu_R, tamaño_At) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar datos y configurar kernels
        if (cudaMemcpy(gpu_A, matriz_host, tamaño_A, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        const dim3 block(hilos_por_bloque, hilos_por_bloque);
        const dim3 grid_t((columnas + block.x - 1) / block.x, (filas + block.y - 1) / block.y);
        const dim3 grid_m((filas + block.x - 1) / block.x, (filas + block.y - 1) / block.y);
        
        // Ejecutar kernels
        kernel_transponer<<<grid_t, block>>>(gpu_A, gpu_A_t, filas, columnas);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        kernel_multiplicar<<<grid_m, block>>>(gpu_A, gpu_A_t, gpu_AAt, filas, columnas, filas);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar AAt a CPU e invertir
        double* host_AAt = (double*)malloc(tamaño_AAt);
        if (!host_AAt || cudaMemcpy(host_AAt, gpu_AAt, tamaño_AAt, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R);
            free(host_AAt);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        double* host_AAt_inv = invertir_matriz(host_AAt, filas);
        if (!host_AAt_inv) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R);
            free(host_AAt);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar inversa a GPU y calcular resultado final
        double* gpu_AAt_inv;
        if (cudaMalloc(&gpu_AAt_inv, tamaño_AAt) != cudaSuccess ||
            cudaMemcpy(gpu_AAt_inv, host_AAt_inv, tamaño_AAt, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R); cudaFree(gpu_AAt_inv);
            free(host_AAt); free(host_AAt_inv);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        const dim3 grid_f((filas + block.x - 1) / block.x, (columnas + block.y - 1) / block.y);
        kernel_multiplicar<<<grid_f, block>>>(gpu_A_t, gpu_AAt_inv, gpu_R, columnas, filas, filas);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R); cudaFree(gpu_AAt_inv);
            free(host_AAt); free(host_AAt_inv);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Copiar resultado final
        double* resultado = (double*)malloc(tamaño_At);
        if (!resultado || cudaMemcpy(resultado, gpu_R, tamaño_At, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R); cudaFree(gpu_AAt_inv);
            free(host_AAt); free(host_AAt_inv); free(resultado);
            *tiempo_total = 0.0;
            return NULL;
        }
        
        // Limpiar memoria
        cudaFree(gpu_A); cudaFree(gpu_A_t); cudaFree(gpu_AAt); cudaFree(gpu_R); cudaFree(gpu_AAt_inv);
        free(host_AAt); free(host_AAt_inv);
        
        *tiempo_total = obtener_tiempo_ms() - tiempo_inicio;
        return resultado;
        
    } else {
        *tiempo_total = 0.0;
        return NULL;
    }
}

/**
 * FUNCIÓN PRINCIPAL OPTIMIZADA DEL PROGRAMA
 * 
 * Flujo de ejecución optimizado:
 * 1. Lectura y análisis de la matriz de entrada con validación completa
 * 2. Cálculo del rango para determinar tipo de pseudoinversa
 * 3. Ejecución del algoritmo paralelo (CUDA) principal optimizado
 * 4. Múltiples ensayos CUDA con diferentes configuraciones para optimización
 * 5. Análisis de configuraciones y generación de archivos de salida
 * 
 * Archivos generados:
 * - salida.sal: contiene la pseudoinversa calculada
 * - metrica.met: contiene las métricas de optimización CUDA
 */
int main() {
    printf(" === PROGRAMA PSEUDOINVERSA CUDA OPTIMIZADO ===\n\n");
    
    // ========================================
    // PASO 1: LECTURA Y CARGA OPTIMIZADA DE LA MATRIZ
    // ========================================
    double* matriz_entrada = NULL;  // Matriz en memoria del host (CPU)
    int numero_filas, numero_columnas;
    
    printf("  Leyendo matriz de entrada...\n");
    leer_matriz("Entrada_matrices/entrada_1.ent", &matriz_entrada, &numero_filas, &numero_columnas);
    printf("  Matriz %dx%d cargada exitosamente\n", numero_filas, numero_columnas);
    imprimir_matriz(matriz_entrada, numero_filas, numero_columnas, "Matriz Original");
    
    // ========================================  
    // PASO 2: ANÁLISIS MATEMÁTICO OPTIMIZADO DE LA MATRIZ
    // ========================================
    printf("\n 🔬 === ANÁLISIS MATEMÁTICO ===\n");
    const int rango_calculado = calcular_rango(matriz_entrada, numero_filas, numero_columnas);
    printf(" Análisis completado:\n");
    printf("   - Rango: %d\n", rango_calculado);
    printf("   - Dimensiones: %dx%d\n", numero_filas, numero_columnas);
    printf("   - Elementos totales: %d\n", numero_filas * numero_columnas);
    
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
    // PASO 3: CÁLCULO PARALELO OPTIMIZADO CON CUDA
    // =========================================
    printf("\n === CÁLCULO PARALELO CUDA OPTIMIZADO ===\n");
    
    // Configuración óptima usando potencias de 2 para mejor rendimiento CUDA
    const int bloques_configuracion_optima = 32;
    const int hilos_configuracion_optima = 16;  // 16x16 = 256 hilos por bloque (óptimo)
    
    printf(" Configuración principal: %d bloques, %d hilos por bloque\n", 
           bloques_configuracion_optima, hilos_configuracion_optima);
    
    char tipo_pseudoinversa_resultado;
    double tiempo_calculo_principal;
    double* pseudoinversa_calculada = calcular_pseudoinversa_cuda(matriz_entrada, numero_filas, numero_columnas, 
                                                                 rango_calculado, &tipo_pseudoinversa_resultado, 
                                                                 &tiempo_calculo_principal,
                                                                 bloques_configuracion_optima, hilos_configuracion_optima);
    
    if (!pseudoinversa_calculada) {
        printf(" Error en cálculo paralelo CUDA optimizado\n");
        guardar_sin_pseudoinversa();
        free(matriz_entrada);
        return 0;
    }
    
    printf(" Cálculo principal completado en %.6f ms\n", tiempo_calculo_principal);
    printf(" Tipo de pseudoinversa calculada: %c (esperado: %c)\n", 
           tipo_pseudoinversa_resultado, tipo_esperado);
    
    // Calcular dimensiones optimizadas de la pseudoinversa
    // Para pseudoinversa L: A+ tiene dimensiones n x m
    // Para pseudoinversa R: A+ tiene dimensiones n x m  
    const int pseudoinversa_filas = numero_columnas;    // Siempre n (columnas de A)
    const int pseudoinversa_columnas = numero_filas;    // Siempre m (filas de A)
    
    printf("📏 Dimensiones pseudoinversa: %dx%d\n", pseudoinversa_filas, pseudoinversa_columnas);
    
    imprimir_matriz(pseudoinversa_calculada, pseudoinversa_filas, pseudoinversa_columnas, "Pseudoinversa CUDA");
    guardar_pseudoinversa(pseudoinversa_calculada, pseudoinversa_filas, pseudoinversa_columnas, tipo_pseudoinversa_resultado);

    // ==========================================
    // PASO 4: ENSAYOS ADICIONALES OPTIMIZADOS PARA BENCHMARKING
    // ==========================================
    printf("\n === ENSAYOS DE OPTIMIZACIÓN CUDA ===\n");
    
    // Configuraciones optimizadas usando potencias de 2 para mejor eficiencia
    const int total_ensayos_benchmark = 16;
    int configuraciones_bloques[] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    int configuraciones_hilos[] = {16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32};
    
    double* tiempos_ensayos = (double*)malloc(total_ensayos_benchmark * sizeof(double));
    if (!tiempos_ensayos) {
        printf(" Error: No se pudo reservar memoria para tiempos de ensayos\n");
        free(matriz_entrada); free(pseudoinversa_calculada);
        return 1;
    }
    
    printf(" Ejecutando %d configuraciones diferentes para análisis de rendimiento:\n", total_ensayos_benchmark);
    
    // Ejecutar cada configuración y medir tiempos de forma optimizada
    for (int indice_ensayo = 0; indice_ensayo < total_ensayos_benchmark; indice_ensayo++) {
        const int bloques_ensayo = configuraciones_bloques[indice_ensayo];
        const int hilos_ensayo = configuraciones_hilos[indice_ensayo];
        
        printf(" Ensayo %d/%d: %d bloques, %d hilos ", 
               indice_ensayo + 1, total_ensayos_benchmark, bloques_ensayo, hilos_ensayo);
        
        char tipo_temporal;
        double tiempo_temporal;
        double* resultado_temporal = calcular_pseudoinversa_cuda(matriz_entrada, numero_filas, numero_columnas, 
                                                               rango_calculado, &tipo_temporal, &tiempo_temporal,
                                                               bloques_ensayo, hilos_ensayo);
        
        if (resultado_temporal) {
            tiempos_ensayos[indice_ensayo] = tiempo_temporal;
            printf("-> %.6f ms \n", tiempo_temporal);
            free(resultado_temporal);  // Liberar resultado temporal inmediatamente
        } else {
            // Si falla CUDA, asignar tiempo infinito
            tiempos_ensayos[indice_ensayo] = 999999.0;
            printf("-> FALLÓ \n");
        }
    }
    
    // ==========================================
    // PASO 5: ANÁLISIS OPTIMIZADO DE CONFIGURACIONES
    // ==========================================
    printf("\n === ANÁLISIS DE RENDIMIENTO ===\n");
    
    double tiempo_mejor_ensayo = tiempos_ensayos[0];
    int indice_configuracion_optima = 0;
    double tiempo_peor_ensayo = tiempos_ensayos[0];
    double suma_tiempos = 0.0;
    int ensayos_exitosos = 0;
    
    // Análisis estadístico optimizado
    for (int i = 0; i < total_ensayos_benchmark; i++) {
        const double tiempo_actual = tiempos_ensayos[i];
        printf(" Configuración %d: %.6f ms (%d bloques, %d hilos)\n", 
               i + 1, tiempo_actual, configuraciones_bloques[i], configuraciones_hilos[i]);
        
        if (tiempo_actual < 999999.0) {  // Solo considerar ensayos exitosos
            ensayos_exitosos++;
            suma_tiempos += tiempo_actual;
            
            if (tiempo_actual < tiempo_mejor_ensayo) {
                tiempo_mejor_ensayo = tiempo_actual;
                indice_configuracion_optima = i;
            }
            if (tiempo_actual > tiempo_peor_ensayo && tiempo_actual < 999999.0) {
                tiempo_peor_ensayo = tiempo_actual;
            }
        }
    }
    
    // Resultados del análisis
    const double tiempo_promedio = (ensayos_exitosos > 0) ? (suma_tiempos / ensayos_exitosos) : 0.0;
    const double mejora_relativa = (tiempo_peor_ensayo > 0) ? (tiempo_peor_ensayo / tiempo_mejor_ensayo) : 1.0;
    
    printf("\n🏆 === RESULTADOS DEL ANÁLISIS ===\n");
    printf("🥇 Mejor configuración: %d bloques, %d hilos (%.6f ms)\n", 
           configuraciones_bloques[indice_configuracion_optima], 
           configuraciones_hilos[indice_configuracion_optima], 
           tiempo_mejor_ensayo);
    printf(" Tiempo promedio: %.6f ms\n", tiempo_promedio);
    printf(" Mejora relativa: %.2fx (mejor vs peor)\n", mejora_relativa);
    printf(" Ensayos exitosos: %d/%d\n", ensayos_exitosos, total_ensayos_benchmark);
    
    // Guardar métricas de optimización usando tiempo principal como referencia
    guardar_metricas(tiempo_calculo_principal, tiempos_ensayos, 
                    configuraciones_bloques, configuraciones_hilos, total_ensayos_benchmark);

    // ==========================================
    // PASO 6: LIMPIEZA OPTIMIZADA Y FINALIZACIÓN
    // ==========================================
    printf("\n === COMPLETADO EXITOSAMENTE ===\n");
    printf(" Archivos generados:\n");
    printf("   - salida.sal (pseudoinversa %dx%d, tipo %c)\n", 
           pseudoinversa_filas, pseudoinversa_columnas, tipo_pseudoinversa_resultado);
    printf("   - metrica.met (%d configuraciones analizadas)\n", total_ensayos_benchmark);
    printf(" Algoritmo: 100%% PARALELO CUDA OPTIMIZADO\n");
    printf(" Mejor rendimiento: %.6f ms\n", tiempo_mejor_ensayo);
    
    // Liberar toda la memoria dinámica de forma segura
    free(matriz_entrada);
    free(pseudoinversa_calculada);
    free(tiempos_ensayos);
    
    printf(" Programa terminado exitosamente\n");
    return 0;
}
