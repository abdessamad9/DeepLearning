using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Common.Layers;
namespace NeuralNetwork.Layers
{
    class L2Penalty : ILayer
    {
        public L2Penalty(int layerSize, int inputSize, int batchSize, IActivator activator, Matrix<double> bias, Matrix<double> activation, Matrix<double> weightedError, IGradientAdjustmentParameters gradientAdjustmentParameters, double penaltyCoefficient)
        {
            LayerSize = layerSize;
            InputSize = inputSize;
            BatchSize = batchSize;
            Activator = activator;
            Bias = bias;
            Activation = activation;
            WeightedError = weightedError;
            GradientAdjustmentParameters = gradientAdjustmentParameters;
            PenaltyCoefficient = penaltyCoefficient;
        }

        public double PenaltyCoefficient
        {
            get
            {
                return PenaltyCoefficient;
            }
            set
            {
                PenaltyCoefficient = value;
            }
        }

        public IGradientAdjustmentParameters GradientAdjustmentParameters
        {
            get
            {
                return GradientAdjustmentParameters;
            }
            set
            {
                GradientAdjustmentParameters = value;
            }
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
        }
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> zeta = WeightedError.Transpose() * Activation + Bias;
            zeta.Map(Activator.ApplyDerivative);
            Vector<double> res;
            upstreamWeightedErrors.Multiply(WeightedError);
            upstreamWeightedErrors.PointwiseMultiply(zeta);
            switch (GradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    WeightedError.Multiply(1-((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate * PenaltyCoefficient);
                    WeightedError.Add(-((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate * Activation * upstreamWeightedErrors.Transpose());
                    res = upstreamWeightedErrors.RowSums();
                    Bias.Add(-((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate * res.ToColumnMatrix());
                    break;

                case GradientAdjustmentType.Adam:
                    throw new NotImplementedException();
                    break;

                case GradientAdjustmentType.Momentum:
                    WeightedError.Multiply(1 - ((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate * PenaltyCoefficient);
                    WeightedError.Multiply(-((MomentumParameters)(GradientAdjustmentParameters)).Momentum);
                    WeightedError.Add(-((MomentumParameters)(GradientAdjustmentParameters)).LearningRate  * Activation * upstreamWeightedErrors.Transpose());
                    res = upstreamWeightedErrors.RowSums();
                    Bias.Multiply(-((MomentumParameters)(GradientAdjustmentParameters)).Momentum);
                    Bias.Add(-((MomentumParameters)(GradientAdjustmentParameters)).LearningRate * res.ToColumnMatrix());
                    break;

                default:
                    throw new InvalidOperationException("Unknown gradient accelerator parameter");
            }

        }

    }
}
