using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Common.Activators;
namespace NeuralNetwork.Common.Layers
{
    public class Standard:ILayer
    {
        public Standard(int layerSize, int inputSize, int batchSize, IActivator activator, Matrix<double> bias, Matrix<double> activation, Matrix<double> weightedError)
        {
            LayerSize = layerSize;
            InputSize = inputSize;
            BatchSize = batchSize;
            Activator = activator;
            Bias = bias;
            Activation = activation;
            WeightedError = weightedError;
        }


        public int LayerSize
        {
            get
            {
                return LayerSize;
            }
            set
            {
                this.LayerSize = value;
            }
        }
        public int InputSize
        {
            get
            {
                return InputSize;
            }
            set
            {
                this.InputSize = value;
            }
        }

        public IActivator Activator
        {
            get
            {
                return Activator;
            }
            set
            {
                this.Activator = value;
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
            set
            {
                this.WeightedError = value;
            }
        }

        public LayerType Type => LayerType.Standard;

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> zeta = WeightedError.Transpose() * Activation + Bias;
            zeta.Map(Activator.ApplyDerivative);

            upstreamWeightedErrors.Multiply(WeightedError);
            upstreamWeightedErrors.PointwiseMultiply(zeta);
            UpdateParameters(upstreamWeightedErrors);


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
            /* Matrix<double> upstreamWeightedErrors = null;


             WeightedError.Add(-0.1 / BatchSize * Activation * upstreamWeightedErrors.Transpose());
             Bias.Add(-0.1 / BatchSize * upstreamWeightedErrors.RowSums());*/
        }

        public void UpdateParameters(Matrix<double> upstreamWeightedErrors)
        {

            WeightedError.Add(-0.1 / BatchSize * Activation * upstreamWeightedErrors.Transpose());
            Vector<double> res = upstreamWeightedErrors.RowSums();
            Bias.Add(-0.1 / BatchSize * res.ToColumnMatrix());
        }
    }
}
