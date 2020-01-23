﻿using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentsParameters;

namespace NeuralNetwork.Layers
{
    class InputStandardizing : ILayer
    {
        public InputStandardizing(int layerSize, int inputSize, int batchSize, IActivator activator, Matrix<double> bias, Matrix<double> activation, Matrix<double> weightedError)
        {
            LayerSize = layerSize;
            InputSize = inputSize;
            BatchSize = batchSize;
            Activator = activator;
            Bias = bias;
            Activation = activation;
            WeightedError = weightedError;
        }

        public IGradientAdjustmentParameters gradientAdjustmentParameters
        {
            get
            {
                return gradientAdjustmentParameters;
            }
            set
            {
                this.gradientAdjustmentParameters = value;
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

        public void UpdateParameters(Matrix<double>  upstreamWeightedErrors )
        {
            switch (gradientAdjustmentParameters.Type)
            {
                case (int)GradientAdjustmentType.FixedLearningRate:

                    //asset = new FixedLearningRateParameters();
                    break;

                case (int)GradientAdjustmentType.Adam:
                    //asset = new AdamParameters();
                    break;

                case (int)GradientAdjustmentType.Momentum:
                    // = new MomentumParameters();
                    break;

                default:
                    throw new InvalidOperationException("Unknown gradient accelerator parameter");
            }


            WeightedError.Add(-0.1 / BatchSize * Activation * upstreamWeightedErrors.Transpose());
             Vector<double> res = upstreamWeightedErrors.RowSums();
             Bias.Add(-0.1 / BatchSize *res.ToColumnMatrix());
        }
    }
}