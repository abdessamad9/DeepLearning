using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Common.Layers
{
    class Dropout:ILayer
    {
        public Dropout(int LayerSize, int InputSize, int BatchSize, IActivator Activator, Matrix<double> Bias, Matrix<double> Activation, Matrix<double> WeightedError)
        {
            LayerSize = LayerSize;
            InputSize = InputSize;
            BatchSize = BatchSize;
            Activator = Activator;
            Bias = Bias;
            Activation = Activation;
            WeightedError = WeightedError;
        }
        public int LayerSize
        {
            get
            {
                return LayerSize;
            }
        }
        public int InputSize
        {
            get
            {
                return InputSize;
            }
        }

        public Identity Activator
        {
            get
            {
                return Activator;
            }
        }

        public Matrix<double> Bias
        {
            get
            {
                return Bias;
            }
            set
            {
                this.Bias = value;
            }
        }


        public int BatchSize
        {
            get
            {
                return BatchSize;
            }
            set
            {
                this.BatchSize = value;
            }
        }

        public Matrix<double> Activation
        {
            get
            {
                return Activation;
            }
            set
            {
                this.Activation = value;
            }
        }

        public Matrix<double> WeightedError
        {
            get
            {
                return WeightedError;
            }
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> zeta = WeightedError.Transpose() * input + Bias;
            zeta.map(Activator.ApplyDerivative);
            PointwiseMultiply(zeta, upstreamWeightedErrors);
            UpdateParameters();


        }

        public bool Equals(ILayer other)
        {
            return WeightedError == other.WeightedError && Activation == other.Activation && LayerSize == other.LayerSize && InputSize == other.InputSize && BatchSize == other.BatchSize;
        }

        public void Propagate(Matrix<double> input)
        {
            System.Diagnostics.Debug.Assert(input.RowCount == 1 && input.ColumnCount == LayerSize, "Dimensions incompatibles");

            Activation = WeightedError.Transpose() * input + Bias;
            Activation.Map(Activator.Apply);
        }

        public void UpdateParameters()
        {
            WeightedError.Add(-0.1 / BatchSize * Activation * upstreamWeightedErrors.Transpose());
            Bias.Add(-0.1 / BatchSize * upstreamWeightedErrors.RowSums());
        }
    }
}
