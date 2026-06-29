# API reference

## Package overview

```{eval-rst}
.. automodule:: xeos
   :no-members:
```

## Model-selector entry points

```{eval-rst}
.. autofunction:: xeos.from_model
.. autofunction:: xeos.equation_of_state
```

`xeos.MODEL_SELECTORS` is a nested mapping
`{model -> {selector string -> canonical EOS id}}` consulted by
{func}`xeos.from_model`. Selectors are matched case-insensitively with
surrounding whitespace stripped.

## Functional shims

```{eval-rst}
.. autofunction:: xeos.rho
.. autofunction:: xeos.alpha
.. autofunction:: xeos.beta
.. autofunction:: xeos.specific_volume
```

## The equation-of-state facade

```{eval-rst}
.. autoclass:: xeos.EquationOfState
   :members:
```

## Registry

```{eval-rst}
.. autofunction:: xeos.list_eos
.. autofunction:: xeos.get_backend
.. autoclass:: xeos.registry.EOSBackend
   :members:
```

## Conventions

```{eval-rst}
.. autoclass:: xeos.TemperatureKind
   :members:
.. autoclass:: xeos.SalinityKind
   :members:
.. autoclass:: xeos.PressureUnit
   :members:
.. automodule:: xeos.conventions
   :members: to_conservative_temperature, to_absolute_salinity, pressure_from_depth
```
