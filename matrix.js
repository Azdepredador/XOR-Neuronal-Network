const random = require('random');
class Matrix {
    /**
     * Rows and columns required for create the matrix.
     * @param {int} r 
     * @param {int} c 
     */
    constructor(r, c, data = null) {
        this.rows = r;
        this.cols = c;
        this.matrix = Array(this.rows).fill().map(() => Array(this.cols).fill(0));
        (data == null) ? this.randomize(): this.fillMatrix(data, r, c);
    }

    /**
     * Create matrix with random values.
     */
    randomize() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] = random.float(-1, 1);
                // this.matrix[i][j] = 3 -1 1;
            }
        }
    }

    /**
     * Fill the matrix with the values readed.
     * @param {matrix} data 
     * @param {int} rows 
     * @param {int} cols 
     */
    fillMatrix(data, rows, cols) {
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                this.matrix[i][j] = data[i][j];
            }
        }
    }

    /**
     * Matrix multiply.
     * @param {Matrix} r - another matrix for multuply 
     */
    multiply(r) {
        let result = new Matrix(this.rows, r.cols);

        if (this.cols == r.rows) {

            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < r.cols; j++) {
                    let sum = 0;
                    for (let k = 0; k < this.cols; k++) {
                        sum += this.matrix[i][k] * r.matrix[k][j];
                    }
                    result.matrix[i][j] = sum;
                }
            }
        } else {
            //Error size matrix
            console.log('Error length matrix')
            result = 0;

        }
        return result;
    }

    /**
     * Convert the array to matrix.
     * @param {Array} arr 
     */
    matrixFromArray(arr) {
        let n = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            n.matrix[i][0] = arr[i];
        }
        return n;
    }

    /**
     * Convert matrix to array.
     */
    toArray() {
        let arr = [this.rows * this.cols];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr[j + i * this.cols] = this.matrix[i][j];
            }
        }
        return arr;
    }

    /**
     * Add the matrix.
     * @param {Matrix} n 
     */
    add(n) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] += n.matrix[i][j];
            }
        }
    }

    /**
     * Substract the matrix
     * @param {Matrix} n 
     */
    substract(n) {
        let result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.matrix[i][j] = this.matrix[i][j] - n.matrix[i][j];
            }
        }
        return result;
    }

    /**
     * Add sigmoid to the matrix.
     */
    addFunctionActivation() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] = this.functionActivation(this.matrix[i][j]);
            }
        }
    }

    /**
     * Transpose of matrix.
     */
    transpose() {
        let trans = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                trans.matrix[j][i] = this.matrix[i][j];
            }
        }
        return trans;
    }

    /**
     * Derivate matrix using dsigmoid.
     */
    derivateFunction() {
        let result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.matrix[i][j] = this.dFunctionActivation(this.matrix[i][j]);
            }
        }
        return result;
    }

    /**
     * Multiply matrix with the learning rate.
     * @param {double} n 
     */
    multiplyLearningRate(n) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] *= n;
            }
        }
    }

    /**
     * Get gradient from matrix
     * @param {Matrix} output 
     * @param {double} learningRate 
     */
    getGradient(output, learningRate) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.matrix[i][j] *= output.matrix[i][j] * learningRate;
            }
        }
    }

    /**
     * Function activation
     * @param {double} x 
     */
    functionActivation(x) {
        //return 1 / (1 + Math.exp(-x));
        return ((x < 0) ? 0 : x);
    }

    /**
     * Function derivate activation
     * @param {double} y 
     */
    dFunctionActivation(y) {
        //return y * (1 - y);
        return ((y < 0) ? 0 : 1);
    }

}

module.exports = Matrix;