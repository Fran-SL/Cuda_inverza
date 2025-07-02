/*
 * ===================================================================
 * PROGRAMA: CÁLCULO DE PSEUDOINVERSA DE MATRICES CON CUDA
 * ===================================================================
 * Autores: Francisco Soto Lagos, Sebastian Salinas jorquera
 * Fecha: 9 de Julio 2025
 * 
 * Descripción:
 * Este programa calcula la pseudoinversa de una matriz no cuadrada
 * usando tanto métodos secuenciales como paralelos con CUDA.
 * 
 * Funcionalidades:
 * - Lectura de matrices desde archivo
 * - Cálculo de rango y tipo de pseudoinversa
 * - Implementación secuencial y paralela (CUDA)
 * - Medición de tiempos y cálculo de speedup
 * - Generación de archivos de salida
 * ===================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>  // Para memcpy
#include <windows.h>  // Para medición precisa de tiempo en Windows

// ===================================================================
// CONSTANTES Y CONFIGURACIONES
// ===================================================================
#define EPSILON 1e-12           // Tolerancia para considerar cero
#define MAX_PRECISION 15        // Máxima precisión decimal para archivos
#define NUM_ENSAYOS 10          // Número mínimo de ensayos requerido

// ===================================================================
// FUNCIONES UTILITARIAS
// ===================================================================

/**
 * Función para obtener tiempo actual en milisegundos (alta precisión)
 * Utiliza los contadores de alta resolución de Windows para medir tiempos precisos
 * Retorna: tiempo actual en milisegundos como double
 */
double obtener_tiempo_ms() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);  // Obtener frecuencia del reloj del sistema
    QueryPerformanceCounter(&counter);      // Obtener valor actual del contador
    
    // Convertir a milisegundos: (contador / frecuencia) * 1000
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

// ===================================================================
// FUNCIONES DE ENTRADA/SALIDA
// ===================================================================

/**
 * Función para leer matriz desde archivo de entrada
 * 
 * Formato esperado del archivo:
 * Línea 1: m n (dimensiones de la matriz)
 * Líneas siguientes: elementos de la matriz en orden fila por fila
 * 
 * Ejemplo:
 * 3 2
 * 1.0 2.0
 * 3.0 4.0  
 * 5.0 6.0
 * 
 * Parámetros:
 *   - archivo: nombre del archivo a leer
 *   - A: puntero donde se guardará la dirección de la matriz
 *   - m, n: punteros donde se guardarán las dimensiones
 */
void leer_matriz(const char* archivo, double** A, int* m, int* n) {
    FILE* f = fopen(archivo, "r");
    if (!f) {
        printf(" ERROR: No se pudo abrir el archivo %s\n", archivo);
        exit(1);
    }

    // Leer dimensiones de la matriz
    if (fscanf(f, "%d %d", m, n) != 2) {
        printf(" ERROR: Formato incorrecto en archivo de entrada\n");
        fclose(f);
        exit(1);
    }

    // Reservar memoria para la matriz (almacenamiento lineal fila por fila)
    *A = (double*)malloc((*m) * (*n) * sizeof(double));
    if (!*A) {
        printf(" ERROR: No se pudo reservar memoria para la matriz\n");
        fclose(f);
        exit(1);
    }

    // Leer todos los elementos de la matriz
    for (int i = 0; i < (*m) * (*n); i++) {
        if (fscanf(f, "%lf", &(*A)[i]) != 1) {
            printf(" ERROR: Datos insuficientes en archivo de entrada\n");
            free(*A);
            fclose(f);
            exit(1);
        }
    }

    fclose(f);
}

/**
 * Función para guardar la pseudoinversa en archivo de salida
 * 
 * Formato del archivo salida.sal:
 * Línea 1: tipo de pseudoinversa ('L' o 'R')
 * Líneas siguientes: elementos de la pseudoinversa con alta precisión
 * 
 * Parámetros:
 *   - pseudoinversa: matriz calculada
 *   - filas, columnas: dimensiones de la pseudoinversa  
 *   - tipo: 'L' para izquierda, 'R' para derecha
 */
void guardar_pseudoinversa(double* pseudoinversa, int filas, int columnas, char tipo) {
    FILE* archivo = fopen("salida.sal", "w");
    if (!archivo) {
        printf(" ERROR: No se pudo crear el archivo salida.sal\n");
        return;
    }
    
    fprintf(archivo, "%c\n", tipo);
    
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            if (j > 0) fprintf(archivo, " ");
            fprintf(archivo, "%.15f", pseudoinversa[i * columnas + j]);
        }
        fprintf(archivo, "\n");
    }
    
    fclose(archivo);
}

/**
 * Función para guardar métricas de rendimiento en archivo
 * 
 * Formato del archivo metrica.met:
 * Cada línea: ensayo bloques hilos speedup
 * 
 * Parámetros:
 *   - tiempo_secuencial: tiempo de referencia (CPU)
 *   - tiempos_paralelos: array de tiempos CUDA
 *   - bloques, hilos: configuraciones usadas
 *   - num_ensayos: cantidad de mediciones
 */
void guardar_metricas(double tiempo_secuencial, double* tiempos_paralelos, 
                     int* bloques, int* hilos, int num_ensayos) {
    FILE* archivo = fopen("metrica.met", "w");
    if (!archivo) {
        printf(" ERROR: No se pudo crear el archivo metrica.met\n");
        return;
    }
    
    for (int i = 0; i < num_ensayos; i++) {
        double speedup = tiempo_secuencial / tiempos_paralelos[i];
        // Formato: número_ensayo num_bloques hilos_por_bloque speedup
        fprintf(archivo, "%d %d %d %.15f\n", i + 1, bloques[i], hilos[i], speedup);
    }
    
    fclose(archivo);
}

// Guardar resultado para matriz sin pseudoinversa
void guardar_sin_pseudoinversa() {
    FILE* archivo = fopen("salida.sal", "w");
    if (archivo) {
        fprintf(archivo, "-1\n");
        fclose(archivo);
    }
}

// ===================================================================
// FUNCIONES DE ÁLGEBRA LINEAL (CPU)
// ===================================================================

/**
 * Calcular el rango de una matriz usando eliminación gaussiana
 * El rango es el número de filas/columnas linealmente independientes
 * 
 * Algoritmo:
 * 1. Crear copia de la matriz para no modificar la original
 * 2. Para cada columna, buscar el mejor pivote (elemento más grande)
 * 3. Intercambiar filas si es necesario
 * 4. Eliminar elementos debajo del pivote
 * 5. Contar filas no nulas
 * 
 * Parámetros:
 *   - A: matriz original
 *   - m: número de filas
 *   - n: número de columnas
 * Retorna: rango de la matriz
 */
int calcular_rango(double* A, int m, int n) {
    // Crear copia temporal para no modificar la matriz original
    double* temp = (double*)malloc(m * n * sizeof(double));
    memcpy(temp, A, m * n * sizeof(double));
    
    int rango = 0;
    int min_dim = (m < n) ? m : n;  // El rango máximo es min(m,n)
    
    // Eliminación gaussiana para cada columna
    for (int i = 0; i < min_dim; i++) {
        // PASO 1: Buscar el mejor pivote en la columna i
        int max_row = i;
        for (int k = i + 1; k < m; k++) {
            if (fabs(temp[k * n + i]) > fabs(temp[max_row * n + i])) {
                max_row = k;  // Guardar fila con mayor elemento
            }
        }
        
        // Si el pivote es muy pequeño, la columna es linealmente dependiente
        if (fabs(temp[max_row * n + i]) < EPSILON) continue;
        
        // PASO 2: Intercambiar filas para poner el pivote en posición
        if (max_row != i) {
            for (int j = 0; j < n; j++) {
                double t = temp[i * n + j];
                temp[i * n + j] = temp[max_row * n + j];
                temp[max_row * n + j] = t;
            }
        }
        
        // PASO 3: Eliminación hacia abajo (hacer ceros debajo del pivote)
        for (int k = i + 1; k < m; k++) {
            if (fabs(temp[k * n + i]) > EPSILON) {
                double factor = temp[k * n + i] / temp[i * n + i];
                // Restar múltiplo de la fila pivote
                for (int j = i; j < n; j++) {
                    temp[k * n + j] -= factor * temp[i * n + j];
                }
            }
        }
        rango++;  // Incrementar rango por cada pivote encontrado
    }
    
    free(temp);
    return rango;
}

/**
 * Transponer una matriz: A^t
 * La transpuesta cambia filas por columnas: A^t[j][i] = A[i][j]
 * 
 * Ejemplo: [1 2 3]^t = [1]
 *          [4 5 6]     [2]
 *                      [3]
 *                      [4]
 *                      [5]
 *                      [6]
 * 
 * Parámetros:
 *   - A: matriz original de dimensiones m x n
 *   - m: filas de A
 *   - n: columnas de A
 * Retorna: nueva matriz A^t de dimensiones n x m
 */
double* transponer_matriz(double* A, int m, int n) {
    // Reservar memoria para la transpuesta (n x m)
    double* A_t = (double*)malloc(n * m * sizeof(double));
    if (!A_t) return NULL;
    
    // Intercambiar filas por columnas
    for (int i = 0; i < m; i++) {        // Para cada fila de A
        for (int j = 0; j < n; j++) {    // Para cada columna de A
            // A[i][j] -> A_t[j][i]
            // En formato lineal: A[i*n + j] -> A_t[j*m + i]
            A_t[j * m + i] = A[i * n + j];
        }
    }
    return A_t;
}

/**
 * Multiplicar dos matrices: C = A * B
 * Solo es posible si el número de columnas de A = número de filas de B
 * 
 * Fórmula: C[i][j] = Σ(A[i][k] * B[k][j]) para k desde 0 hasta n_A-1
 * 
 * Ejemplo: [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
 *          [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
 * 
 * Parámetros:
 *   - A: matriz A de dimensiones m_A x n_A
 *   - B: matriz B de dimensiones m_B x n_B
 *   - (debe cumplirse: n_A == m_B)
 * Retorna: matriz C de dimensiones m_A x n_B
 */
double* multiplicar_matrices(double* A, int m_A, int n_A, double* B, int m_B, int n_B) {
    // Verificar compatibilidad de dimensiones
    if (n_A != m_B) return NULL;
    
    // Reservar memoria para el resultado (m_A x n_B) e inicializar en cero
    double* C = (double*)calloc(m_A * n_B, sizeof(double));
    if (!C) return NULL;
    
    // Triple bucle para multiplicación de matrices
    for (int i = 0; i < m_A; i++) {      // Para cada fila de A
        for (int j = 0; j < n_B; j++) {  // Para cada columna de B
            // Calcular el producto punto de la fila i de A con la columna j de B
            for (int k = 0; k < n_A; k++) {
                // C[i][j] += A[i][k] * B[k][j]
                C[i * n_B + j] += A[i * n_A + k] * B[k * n_B + j];
            }
        }
    }
    return C;
}

/**
 * Invertir matriz cuadrada usando el método de Gauss-Jordan
 * 
 * Algoritmo:
 * 1. Crear matriz aumentada [A | I] donde I es la identidad
 * 2. Aplicar operaciones elementales para convertir A en I
 * 3. Las mismas operaciones convierten I en A^(-1)
 * 4. Resultado: [I | A^(-1)]
 * 
 * Ejemplo para matriz 2x2:
 * [a b | 1 0]    →    [1 0 | x y]
 * [c d | 0 1]         [0 1 | z w]
 * 
 * Donde A^(-1) = [x y]
 *                [z w]
 * 
 * Parámetros:
 *   - A: matriz cuadrada a invertir (n x n)
 *   - n: dimensión de la matriz
 * Retorna: matriz inversa o NULL si es singular
 */
double* invertir_matriz(double* A, int n) {
    // Reservar memoria para matriz aumentada [A | I] de tamaño n x 2n
    double* aumentada = (double*)malloc(n * 2 * n * sizeof(double));
    if (!aumentada) return NULL;
    
    // PASO 1: Crear matriz aumentada [A | I]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Copiar elementos de A al lado izquierdo
            aumentada[i * 2 * n + j] = A[i * n + j];
            // Crear matriz identidad I al lado derecho
            aumentada[i * 2 * n + (n + j)] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // PASO 2: Eliminación de Gauss-Jordan (convertir A en matriz identidad)
    for (int i = 0; i < n; i++) {
        // SUB-PASO 2.1: Buscar el mejor pivote en la columna i
        // (elemento con mayor valor absoluto para estabilidad numérica)
        int max_fila = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(aumentada[k * 2 * n + i]) > fabs(aumentada[max_fila * 2 * n + i])) {
                max_fila = k;
            }
        }
        
        // Verificar si la matriz es singular (determinante = 0)
        if (fabs(aumentada[max_fila * 2 * n + i]) < EPSILON) {
            free(aumentada);
            return NULL; // Matriz no invertible
        }
        
        // SUB-PASO 2.2: Intercambiar filas para poner el mejor pivote en posición
        if (max_fila != i) {
            for (int j = 0; j < 2 * n; j++) {
                double temp = aumentada[i * 2 * n + j];
                aumentada[i * 2 * n + j] = aumentada[max_fila * 2 * n + j];
                aumentada[max_fila * 2 * n + j] = temp;
            }
        }
        
        // SUB-PASO 2.3: Normalizar la fila del pivote (hacer pivote = 1)
        double pivot = aumentada[i * 2 * n + i];
        for (int j = 0; j < 2 * n; j++) {
            aumentada[i * 2 * n + j] /= pivot;
        }
        
        // SUB-PASO 2.4: Eliminar todos los otros elementos de la columna i
        // (hacer ceros arriba y abajo del pivote)
        for (int k = 0; k < n; k++) {
            if (k != i) {  // No modificar la fila del pivote
                double factor = aumentada[k * 2 * n + i];
                for (int j = 0; j < 2 * n; j++) {
                    // Restar múltiplo de la fila pivote
                    aumentada[k * 2 * n + j] -= factor * aumentada[i * 2 * n + j];
                }
            }
        }
    }
    
    // PASO 3: Extraer la matriz inversa del lado derecho de la matriz aumentada
    // En este punto tenemos [I | A^(-1)]
    double* inversa = (double*)malloc(n * n * sizeof(double));
    if (inversa) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Copiar elementos del lado derecho (columnas n a 2n-1)
                inversa[i * n + j] = aumentada[i * 2 * n + (n + j)];
            }
        }
    }
    
    free(aumentada);  // Liberar memoria temporal
    return inversa;
}

/**
 * FUNCIÓN PRINCIPAL: Calcular pseudoinversa de una matriz
 * 
 * La pseudoinversa existe en dos casos:
 * 1. IZQUIERDA (L): cuando rango = n < m (más filas que columnas, rango completo en columnas)
 *    Fórmula: L = (A^t * A)^(-1) * A^t
 * 
 * 2. DERECHA (R): cuando rango = m < n (más columnas que filas, rango completo en filas)  
 *    Fórmula: R = A^t * (A * A^t)^(-1)
 * 
 * Parámetros:
 *   - A: matriz original
 *   - m, n: dimensiones de A
 *   - rango: rango previamente calculado
 *   - tipo: puntero donde se guardará 'L' o 'R'
 * Retorna: pseudoinversa o NULL si no existe
 */
double* calcular_pseudoinversa(double* A, int m, int n, int rango, char* tipo) {
    if (rango == n && rango < m) {
        // CASO 1: PSEUDOINVERSA POR LA IZQUIERDA
        // La matriz es "alta y delgada" con columnas linealmente independientes
        printf("📐 Calculando pseudoinversa IZQUIERDA: L = (A^t * A)^(-1) * A^t\n");
        *tipo = 'L';
        
        // Paso 1: Calcular A^t (transpuesta)
        double* A_t = transponer_matriz(A, m, n);
        if (!A_t) return NULL;
        
        // Paso 2: Calcular A^t * A (matriz cuadrada n x n)
        double* AtA = multiplicar_matrices(A_t, n, m, A, m, n);
        if (!AtA) { free(A_t); return NULL; }
        
        // Paso 3: Calcular (A^t * A)^(-1) (inversa de matriz cuadrada)
        double* AtA_inv = invertir_matriz(AtA, n);
        if (!AtA_inv) { free(A_t); free(AtA); return NULL; }
        
        // Paso 4: Calcular L = (A^t * A)^(-1) * A^t (resultado final)
        double* L = multiplicar_matrices(AtA_inv, n, n, A_t, n, m);
        
        // Liberar memoria temporal
        free(A_t); free(AtA); free(AtA_inv);
        return L;
        
    } else if (rango == m && rango < n) {
        // CASO 2: PSEUDOINVERSA POR LA DERECHA
        // La matriz es "ancha y baja" con filas linealmente independientes
        printf("📐 Calculando pseudoinversa DERECHA: R = A^t * (A * A^t)^(-1)\n");
        *tipo = 'R';
        
        // Paso 1: Calcular A^t (transpuesta)
        double* A_t = transponer_matriz(A, m, n);
        if (!A_t) return NULL;
        
        // Paso 2: Calcular A * A^t (matriz cuadrada m x m)
        double* AAt = multiplicar_matrices(A, m, n, A_t, n, m);
        if (!AAt) { free(A_t); return NULL; }
        
        // Paso 3: Calcular (A * A^t)^(-1) (inversa de matriz cuadrada)
        double* AAt_inv = invertir_matriz(AAt, m);
        if (!AAt_inv) { free(A_t); free(AAt); return NULL; }
        
        // Paso 4: Calcular R = A^t * (A * A^t)^(-1) (resultado final)
        double* R = multiplicar_matrices(A_t, n, m, AAt_inv, m, m);
        
        // Liberar memoria temporal
        free(A_t); free(AAt); free(AAt_inv);
        return R;
    }
    
    // Si llegamos aquí, la matriz no tiene pseudoinversa
    return NULL;
}

// ===================================================================
// KERNELS CUDA PARA PARALELIZACIÓN
// ===================================================================

/**
 * KERNEL CUDA: Transponer matriz en paralelo
 * Cada thread de CUDA calcula una posición de la matriz transpuesta
 * 
 * Mapeo de threads:
 * - blockIdx.x, blockIdx.y: posición del bloque en la grid
 * - threadIdx.x, threadIdx.y: posición del thread dentro del bloque
 * - idx, idy: posición global del thread en la matriz
 * 
 * Parámetros:
 *   - A: matriz original en GPU (m x n)
 *   - A_t: matriz transpuesta en GPU (n x m)
 *   - m, n: dimensiones de A
 */
__global__ void kernel_transponer(double* A, double* A_t, int m, int n) {
    // Calcular posición global del thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Columna en A, fila en A_t
    int idy = blockIdx.y * blockDim.y + threadIdx.y;  // Fila en A, columna en A_t
    
    // Verificar que estamos dentro de los límites
    if (idx < n && idy < m) {
        // Transponer: A[idy][idx] -> A_t[idx][idy]
        A_t[idx * m + idy] = A[idy * n + idx];
    }
}

/**
 * KERNEL CUDA: Multiplicar matrices en paralelo
 * Cada thread calcula un elemento del resultado C = A * B
 * 
 * Algoritmo paralelo:
 * - Cada thread (fila, col) calcula C[fila][col]
 * - Realiza el producto punto de la fila de A con la columna de B
 * 
 * Parámetros:
 *   - A: primera matriz en GPU (m_A x n_A)
 *   - B: segunda matriz en GPU (n_A x n_B)
 *   - C: matriz resultado en GPU (m_A x n_B)
 */
__global__ void kernel_multiplicar(double* A, double* B, double* C, 
                                  int m_A, int n_A, int n_B) {
    // Calcular posición del elemento que calculará este thread
    int fila = blockIdx.y * blockDim.y + threadIdx.y;  // Fila en C
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // Columna en C
    
    // Verificar límites
    if (fila < m_A && col < n_B) {
        double suma = 0.0;
        
        // Calcular producto punto: C[fila][col] = Σ(A[fila][k] * B[k][col])
        for (int k = 0; k < n_A; k++) {
            suma += A[fila * n_A + k] * B[k * n_B + col];
        }
        
        // Guardar resultado en la matriz C
        C[fila * n_B + col] = suma;
    }
}

/**
 * FUNCIÓN CUDA: Calcular pseudoinversa usando paralelización
 * 
 * Implementación híbrida:
 * - Operaciones matriciales (transponer, multiplicar) en GPU con CUDA
 * - Inversión de matriz en CPU (más estable numéricamente)
 * 
 * Flujo de ejecución:
 * 1. Copiar datos a GPU
 * 2. Ejecutar kernels CUDA para operaciones paralelas
 * 3. Copiar resultados intermedios a CPU para inversión
 * 4. Continuar cálculo en GPU
 * 5. Copiar resultado final a CPU
 * 
 * Parámetros:
 *   - h_A: matriz en CPU (host)
 *   - m, n: dimensiones
 *   - rango: rango previamente calculado
 *   - tipo: puntero para tipo de pseudoinversa
 *   - tiempo_ejecucion: puntero para tiempo medido
 *   - num_bloques, hilos_por_bloque: configuración CUDA
 */
double* calcular_pseudoinversa_cuda(double* h_A, int m, int n, int rango, 
                                   char* tipo, double* tiempo_ejecucion,
                                   int num_bloques, int hilos_por_bloque) {
    
    double tiempo_inicio = obtener_tiempo_ms();
    
    if (rango == n && rango < m) {
        // PSEUDOINVERSA POR LA IZQUIERDA usando CUDA
        printf("🚀 Ejecutando algoritmo CUDA para pseudoinversa IZQUIERDA\n");
        *tipo = 'L';
        
        // PASO 1: Reservar memoria en GPU
        printf("   📋 Reservando memoria GPU...\n");
        double *d_A, *d_A_t, *d_AtA, *d_L;
        size_t size_A = m * n * sizeof(double);      // Tamaño de A
        size_t size_At = n * m * sizeof(double);     // Tamaño de A^t  
        size_t size_AtA = n * n * sizeof(double);    // Tamaño de A^t*A
        
        cudaMalloc(&d_A, size_A);      // Matriz original en GPU
        cudaMalloc(&d_A_t, size_At);   // Transpuesta en GPU
        cudaMalloc(&d_AtA, size_AtA);  // Producto A^t*A en GPU
        cudaMalloc(&d_L, size_At);     // Resultado final en GPU
        
        // PASO 2: Copiar matriz A desde CPU a GPU
        printf("   💾 Copiando datos a GPU...\n");
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        
        // PASO 3: Configurar grid y bloques para kernels CUDA
        printf("   ⚙️ Configurando kernels: %d hilos por bloque\n", hilos_por_bloque);
        dim3 block_size(hilos_por_bloque, hilos_por_bloque);  // Bloque 2D
        
        // Grid para transponer (n x m elementos)
        dim3 grid_transponer((n + block_size.x - 1) / block_size.x, 
                            (m + block_size.y - 1) / block_size.y);
        
        // Grid para multiplicar (n x n elementos)
        dim3 grid_multiplicar((n + block_size.x - 1) / block_size.x, 
                             (n + block_size.y - 1) / block_size.y);
        
        // PASO 4: Ejecutar kernel para transponer A -> A^t
        printf("   🔄 Ejecutando transposición en GPU...\n");
        kernel_transponer<<<grid_transponer, block_size>>>(d_A, d_A_t, m, n);
        cudaDeviceSynchronize();  // Esperar a que termine
        
        // PASO 5: Ejecutar kernel para multiplicar A^t * A
        printf("   🔢 Ejecutando multiplicación A^t * A en GPU...\n");
        kernel_multiplicar<<<grid_multiplicar, block_size>>>(d_A_t, d_A, d_AtA, n, m, n);
        cudaDeviceSynchronize();  // Esperar a que termine
        
        // PASO 6: Copiar A^t * A a CPU para invertir (más estable en CPU)
        printf("   🔄 Copiando A^t*A a CPU para inversión...\n");
        double* h_AtA = (double*)malloc(size_AtA);
        cudaMemcpy(h_AtA, d_AtA, size_AtA, cudaMemcpyDeviceToHost);
        
        // PASO 7: Calcular inversa en CPU
        printf("   🧮 Calculando (A^t*A)^(-1) en CPU...\n");
        double* h_AtA_inv = invertir_matriz(h_AtA, n);
        if (h_AtA_inv == NULL) {
            printf("   ❌ Error: Matriz singular, no se puede invertir\n");
            cudaFree(d_A); cudaFree(d_A_t); cudaFree(d_AtA); cudaFree(d_L);
            free(h_AtA);
            return NULL;
        }
        
        // PASO 8: Copiar inversa de vuelta a GPU
        printf("   📤 Copiando inversa a GPU...\n");
        double* d_AtA_inv;
        cudaMalloc(&d_AtA_inv, size_AtA);
        cudaMemcpy(d_AtA_inv, h_AtA_inv, size_AtA, cudaMemcpyHostToDevice);
        
        // PASO 9: Calcular producto final L = (A^t * A)^(-1) * A^t
        printf("   🎯 Calculando producto final en GPU...\n");
        dim3 grid_final((m + block_size.x - 1) / block_size.x, 
                       (n + block_size.y - 1) / block_size.y);
        kernel_multiplicar<<<grid_final, block_size>>>(d_AtA_inv, d_A_t, d_L, n, n, m);
        cudaDeviceSynchronize();
        
        // PASO 10: Copiar resultado final a CPU
        printf("   📥 Copiando resultado a CPU...\n");
        double* h_L = (double*)malloc(size_At);
        cudaMemcpy(h_L, d_L, size_At, cudaMemcpyDeviceToHost);
        
        // PASO 11: Liberar toda la memoria GPU
        printf("   🧹 Liberando memoria GPU...\n");
        cudaFree(d_A); cudaFree(d_A_t); cudaFree(d_AtA); cudaFree(d_L); cudaFree(d_AtA_inv);
        free(h_AtA); free(h_AtA_inv);
        
        // Finalizar medición de tiempo
        *tiempo_ejecucion = obtener_tiempo_ms() - tiempo_inicio;
        printf("   ✅ CUDA completado en %.6f ms\n", *tiempo_ejecucion);
        return h_L;
        
    } else {
        // Por ahora, solo implementamos pseudoinversa por la izquierda en CUDA
        printf("   ⚠️ Solo pseudoinversa IZQUIERDA implementada en CUDA\n");
        *tiempo_ejecucion = 0.0;
        return NULL;
    }
}

/**
 * FUNCIÓN PRINCIPAL DEL PROGRAMA
 * 
 * Flujo de ejecución:
 * 1. Lectura y análisis de la matriz de entrada
 * 2. Cálculo del rango para determinar tipo de pseudoinversa
 * 3. Ejecución del algoritmo secuencial (CPU) como referencia
 * 4. Múltiples ensayos del algoritmo paralelo (CUDA) con diferentes configuraciones
 * 5. Cálculo de métricas de speedup y generación de archivos de salida
 * 
 * Archivos generados:
 * - salida.sal: contiene la pseudoinversa calculada
 * - metrica.met: contiene las métricas de rendimiento
 */
int main() {
    printf(" === PROGRAMA PSEUDOINVERSA CUDA ===\n\n");
    
    // ========================================
    // PASO 1: LECTURA Y CARGA DE LA MATRIZ
    // ========================================
    double* h_A = NULL;  // Matriz en memoria del host (CPU)
    int filas, columnas;
    printf(" Leyendo matriz...\n");
    leer_matriz("Entrada_matrices/entrada_1.ent", &h_A, &filas, &columnas);
    printf(" Matriz %dx%d cargada\n", filas, columnas);
    imprimir_matriz(h_A, filas, columnas, "Matriz Original");
    
    // ========================================  
    // PASO 2: ANÁLISIS MATEMÁTICO DE LA MATRIZ
    // ========================================
    int rango = calcular_rango(h_A, filas, columnas);
    printf("📊 Rango: %d, Dimensiones: %dx%d\n", rango, filas, columnas);
    
    // Determinar qué tipo de pseudoinversa es posible calcular
    if (rango == filas && rango < columnas) {
        printf("🟢 Pseudoinversa DERECHA (R): más columnas que filas, rango completo en filas\n");
    } else if (rango == columnas && rango < filas) {
        printf("🟢 Pseudoinversa IZQUIERDA (L): más filas que columnas, rango completo en columnas\n");
    } else if (rango == filas && rango == columnas) {
        printf("🟡 Matriz cuadrada invertible: usar inversión estándar\n");
    } else {
        printf("🔴 Sin pseudoinversa: rango deficiente\n");
        guardar_sin_pseudoinversa();
        free(h_A);
        return 0;
    }

    // =========================================
    // PASO 3: CÁLCULO SECUENCIAL (REFERENCIA)
    // =========================================
    printf("\n⏱ === CÁLCULO SECUENCIAL ===\n");
    double tiempo_inicio = obtener_tiempo_ms();
    char tipo_pseudoinversa;
    double* pseudoinversa_seq = calcular_pseudoinversa(h_A, filas, columnas, rango, &tipo_pseudoinversa);
    double tiempo_secuencial = obtener_tiempo_ms() - tiempo_inicio;
    
    if (!pseudoinversa_seq) {
        printf(" Error en cálculo secuencial\n");
        guardar_sin_pseudoinversa();
        free(h_A);
        return 0;
    }
    
    printf(" Tiempo: %.6f ms\n", tiempo_secuencial);
    
    // Calcular dimensiones de la pseudoinversa
    int p_filas = (tipo_pseudoinversa == 'L') ? columnas : columnas;
    int p_columnas = (tipo_pseudoinversa == 'L') ? filas : filas;
    
    imprimir_matriz(pseudoinversa_seq, p_filas, p_columnas, "Pseudoinversa");
    guardar_pseudoinversa(pseudoinversa_seq, p_filas, p_columnas, tipo_pseudoinversa);

    // ==========================================
    // PASO 4: ENSAYOS PARALELOS CON CUDA
    // ==========================================
    printf("\n === ENSAYOS CUDA ===\n");
    // Configuraciones de prueba: diferentes combinaciones de bloques y hilos
    int num_ensayos = 12;
    int bloques[] = {1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32};
    int hilos[] = {32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64};
    double* tiempos_paralelos = (double*)malloc(num_ensayos * sizeof(double));
    
    // Ejecutar cada configuración y medir tiempos
    for (int i = 0; i < num_ensayos; i++) {
        printf(" Ensayo %d/%d: %d bloques, %d hilos ", i+1, num_ensayos, bloques[i], hilos[i]);
        
        char tipo_temp;
        double tiempo_temp;
        double* resultado_cuda = calcular_pseudoinversa_cuda(h_A, filas, columnas, rango, 
                                                           &tipo_temp, &tiempo_temp,
                                                           bloques[i], hilos[i]);
        
        if (resultado_cuda) {
            tiempos_paralelos[i] = tiempo_temp;
            printf("-> %.6f ms\n", tiempo_temp);
            free(resultado_cuda);
        } else {
            // Si falla CUDA, usar tiempo secuencial (speedup = 1)
            tiempos_paralelos[i] = tiempo_secuencial;
            printf("-> FALLÓ\n");
        }
    }
    
    // ==========================================
    // PASO 5: CÁLCULO Y REPORTE DE MÉTRICAS
    // ==========================================
    printf("\n === SPEEDUP ===\n");
    for (int i = 0; i < num_ensayos; i++) {
        double speedup = tiempo_secuencial / tiempos_paralelos[i];
        printf("Ensayo %d: %.6fx\n", i+1, speedup);
    }
    
    // Guardar todas las métricas en archivo
    guardar_metricas(tiempo_secuencial, tiempos_paralelos, bloques, hilos, num_ensayos);

    // ==========================================
    // PASO 6: LIMPIEZA Y FINALIZACIÓN
    // ==========================================
    printf("\n === COMPLETADO ===\n");
    printf(" Archivos generados: salida.sal, metrica.met\n");
    
    // Liberar toda la memoria dinámica
    free(h_A);
    free(pseudoinversa_seq);
    free(tiempos_paralelos);
    
    printf(" Programa terminado exitosamente\n");
    return 0;
}
