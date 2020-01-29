using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Common.Layers;

namespace NeuralNetwork.Layers
{
    public class Standard : ILayer
    {
        //
        public Standard( int batchSize, IActivator activator, double[] bias, double[,] weights, IGradientAdjustmentParameters gradientAdjustmentParameters)
        {

            BatchSize = batchSize;
            LayerSize = bias.Length;
            InputSize = weights.GetLength(0);
            Activator =activator;
            Bias = Matrix<double>.Build.Dense(bias.Length, BatchSize,0);
            for(int i= 0; i < BatchSize; i++)
            {
                Bias.SetColumn(i, bias);
            }
            B = Matrix<double>.Build.Dense(LayerSize, BatchSize,0);
            Weights = DenseMatrix.OfArray(weights);
            WeightedError = Matrix<double>.Build.Dense(Weights.RowCount,B.ColumnCount,0);
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize,0);
            GradientAdjustmentParameters = gradientAdjustmentParameters;
        }

        Matrix<double> lastActivation;
        public Matrix<double> LastActivation
        {
            get
            {
                return lastActivation;
            }
            set
            {
                lastActivation = value;
            }
        }
        Matrix<double> b;
        public Matrix<double> B
        {
            get
            {
                return b;
            }
            set
            {
                b = value;
            }
        }
        Matrix<double> weights;
        public Matrix<double> Weights
        {
            get
            {
                return weights;
            }
            set
            {
                weights = value;
            }
        }
        IGradientAdjustmentParameters gradientAdjustmentParameters;
        public IGradientAdjustmentParameters GradientAdjustmentParameters
        {
            get
            {
                return gradientAdjustmentParameters;
            }
            set
            {
                this.gradientAdjustmentParameters =  value;
            }
        }

        int layerSize;
        public int LayerSize
        {
            get
            {
                return layerSize;
            }
            set
            {
                this.layerSize = value;
            }
        }
        int inputSize;
        public int InputSize
        {
            get
            {
                return inputSize;
            }
            set
            {
                this.inputSize = value;
            }
        }
        IActivator activator;
        public IActivator Activator
        {
            get
            {
                return activator;
            }
            set
            {
                this.activator = value;
            }
        }

        Matrix<double> bias;
        public Matrix<double> Bias
        {
            get
            {
                return bias;
            }
            set
            {
                this.bias = value;
            }
        }

        int batchSize;
        public int BatchSize
        {
            get
            {
                return batchSize;
            }
            set
            {
                this.batchSize = value;
            }
        }
        Matrix<double> zeta;
        public Matrix<double> Zeta
        {
            get
            {
                return zeta;
            }
            set
            {
                this.zeta = value;
            }
        }
        Matrix<double> activation;
        public Matrix<double> Activation
        {
            get
            {
                return activation;
            }
            set
            {
                this.activation = value;
            }
        }

        Matrix<double> weightedError;
        public Matrix<double> WeightedError
        {
            get
            {
                return weightedError;
            }
            set
            {
                this.weightedError = value;
            }
        }
        
        public bool Equals(ILayer other)
        {
            return  LayerSize == other.LayerSize && InputSize == other.InputSize && BatchSize == other.BatchSize;
        }

        public void Propagate(Matrix<double> input)
        {
            LastActivation = input;
            Zeta = weights.Transpose() * input + Bias;
            Zeta.Map(Activator.Apply, Activation);
        }

        public void UpdateParameters()
        {
            //Vector<double> res;
            switch (GradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    var gradW = -((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate / BatchSize * LastActivation * B.Transpose();
                    var gradB = -((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate / BatchSize * B * Vector<double>.Build.Dense(BatchSize, 1);
                    Weights.Add(gradW);
                    for(int i= 0; i < BatchSize; i++)
                    {
                        Bias.SetColumn(i,  Bias.Column(i) + gradB);
                    }
                    break;

                case GradientAdjustmentType.Adam:
                    throw new NotImplementedException();
                    break;

                case GradientAdjustmentType.Momentum:
                    /*WeightedError.Multiply(-((MomentumParameters)(GradientAdjustmentParameters)).Momentum);
                    WeightedError.Add(-((MomentumParameters)(GradientAdjustmentParameters)).LearningRate  * Activation * upstreamWeightedErrors.Transpose());
                     res = upstreamWeightedErrors.RowSums();
                    Bias.Multiply(-((MomentumParameters)(GradientAdjustmentParameters)).Momentum);
                    Bias.Add(-((MomentumParameters)(GradientAdjustmentParameters)).LearningRate  * res.ToColumnMatrix());*/
                    break;

                default:
                    throw new InvalidOperationException("Unknown gradient accelerator parameter");
            }
        }
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> ZetaPrime = Matrix<double>.Build.Dense(LayerSize,BatchSize,0);
            zeta.Map(Activator.ApplyDerivative,ZetaPrime);
            B = ZetaPrime.PointwiseMultiply(upstreamWeightedErrors);
            WeightedError = Weights * B;

        }

       
    }
}
