# CLA_hadamard

This folder contains the implementation of the project “Structure exploiting Lanczos method for Hadamard product of low-rank matrices” for the CLA course.

Main files:

```driver.m```: main program

```matvec_hadamard.m```: implementation of the structure-exploiting matrix vector multiplication

```matmat_hadamard.m```: implementation of the structure-exploiting matrix vector multiplication

```lanczos.m```: implementation of the Lanczos method to approximate eigenpairs of a matrix A

```Afun.m```: implementation of <img src="https://latex.codecogs.com/svg.latex?(\mathbf{C}\mathbf{C}^T)\mathbf{v}" title="(\mathbf{C}\mathbf{C}^T)\mathbf{v}" />
or <img src="https://latex.codecogs.com/svg.latex?(\mathbf{C}\mathbf{C}^T)\mathbf{v}" title="(\mathbf{C}^T\mathbf{C})\mathbf{v}" /> where C is the Hadamard product between two matrices

Last update: 05 June 2019
