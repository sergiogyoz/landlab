#! /usr/bin/env python
"""Unit tests for landlab.components.flexure.Flexure1D."""
import pytest
from nose.tools import with_setup
from numpy.testing import (
    assert_array_equal,
    assert_array_less,
    assert_array_almost_equal,
    assert_almost_equal,
)

import numpy as np

from landlab import RasterModelGrid
from landlab.components.flexure import Flexure1D


(_SHAPE, _SPACING, _ORIGIN) = ((20, 20), (10e3, 10e3), (0., 0.))
_ARGS = (_SHAPE, _SPACING, _ORIGIN)


def setup_grid():
    from landlab import RasterModelGrid

    grid = RasterModelGrid((20, 20), spacing=10e3)
    flex = Flexure1D(grid)
    globals().update({"flex": Flexure1D(grid)})


@with_setup(setup_grid)
def test_name():
    """Test component name exists and is a string."""
    assert isinstance(flex.name, str)


@with_setup(setup_grid)
def test_input_var_names():
    """Test input_var_names is a tuple of strings."""
    assert isinstance(flex.input_var_names, tuple)
    for name in flex.input_var_names:
        assert isinstance(name, str)


@with_setup(setup_grid)
def test_output_var_names():
    """Test output_var_names is a tuple of strings."""
    assert isinstance(flex.output_var_names, tuple)
    for name in flex.output_var_names:
        assert isinstance(name, str)


@with_setup(setup_grid)
def test_var_units():
    """Test input/output var units."""
    assert isinstance(flex.units, tuple)
    for name, units in flex.units:
        assert name in flex.input_var_names + flex.output_var_names
        assert isinstance(units, str)


@with_setup(setup_grid)
def test_var_mapping():
    """Test input/output var mappings."""
    assert isinstance(flex._var_mapping, dict)
    for name in flex.input_var_names + flex.output_var_names:
        assert name in flex._var_mapping
        assert isinstance(flex._var_mapping[name], str)
        assert flex._var_mapping[name] in (
            "node",
            "link",
            "patch",
            "corner",
            "face",
            "cell",
        )


@with_setup(setup_grid)
def test_var_doc():
    """Test input/output var docs."""
    assert isinstance(flex._var_doc, dict)
    for name in flex.input_var_names + flex.output_var_names:
        assert name in flex._var_doc
        assert isinstance(flex._var_doc[name], str)


def test_calc_airy():
    """Test airy isostasy."""
    flex = Flexure1D(RasterModelGrid((3, 5)), method="airy")
    flex.load_at_node[:] = flex.gamma_mantle

    assert_array_equal(flex.dz_at_node, 0.)
    flex.update()
    assert_array_equal(flex.dz_at_node, 1.)


def test_run_one_step():
    """Test the run_one_step method."""
    flex = Flexure1D(RasterModelGrid((3, 5)), method="airy")
    flex.load_at_node[:] = flex.gamma_mantle

    assert_array_equal(flex.dz_at_node, 0.)
    flex.run_one_step()
    assert_array_equal(flex.dz_at_node, 1.)


def test_with_one_row():
    """Test calculating on one row."""
    flex = Flexure1D(RasterModelGrid((3, 5)), method="airy", rows=1)
    flex.load_at_node[:] = flex.gamma_mantle

    assert_array_equal(flex.dz_at_node, 0.)
    flex.update()
    assert_array_equal(flex.dz_at_node[0], 0.)
    assert_array_equal(flex.dz_at_node[1], 1.)
    assert_array_equal(flex.dz_at_node[2], 0.)


def test_with_one_row():
    """Test calculating on one row."""
    flex = Flexure1D(RasterModelGrid((3, 5)), method="airy", rows=(0, 2))
    flex.load_at_node[:] = -flex.gamma_mantle

    assert_array_equal(flex.dz_at_node, 0.)
    flex.update()
    assert_array_equal(flex.dz_at_node[0], -1.)
    assert_array_equal(flex.dz_at_node[1], 0.)
    assert_array_equal(flex.dz_at_node[2], -1.)


def test_field_is_updated():
    """Test the output field is updated."""
    flex = Flexure1D(RasterModelGrid((3, 5)), method="airy", rows=(0, 2))
    flex.load_at_node[:] = -flex.gamma_mantle

    assert_array_equal(flex.dz_at_node, 0.)
    flex.update()

    dz = flex.grid.at_node["lithosphere_surface__increment_of_elevation"]
    assert_array_equal(flex.dz_at_node.flatten(), dz)


def test_calc_flexure():
    """Test calc_flexure function."""
    x = np.arange(100.)
    loads = np.ones(100)
    dz = Flexure1D.calc_flexure(x, loads, 1., 1.)

    assert_array_less(0., dz)
    assert isinstance(dz, np.ndarray)
    assert dz.shape == loads.shape
    assert dz.dtype == loads.dtype


def test_calc_flexure_with_out_keyword():
    """Test calc_flexure out keyword."""
    x = np.arange(100.)
    loads = np.ones(100)
    buffer = np.empty_like(x)
    dz = Flexure1D.calc_flexure(x, loads, 1., 1., out=buffer)
    assert np.may_share_memory(dz, buffer)


def test_calc_flexure():
    """Test calc_flexure with multiple rows of loads."""
    x = np.arange(100.) * 1e3
    loads = np.ones(500).reshape((5, 100))
    dz = Flexure1D.calc_flexure(x, loads, 1e4, 1.)

    assert_array_less(0., dz)
    assert isinstance(dz, np.ndarray)
    assert dz.shape == loads.shape
    assert dz.dtype == loads.dtype

    for row in range(5):
        assert_array_almost_equal(dz[0], dz[row])


def test_setter_updates():
    """Test that the setters update dependant parameters."""
    setters = {
        "eet": ("rigidity", "alpha"),
        "youngs": ("rigidity", "alpha"),
        "rho_water": ("gamma_mantle", "alpha"),
        "rho_mantle": ("gamma_mantle", "alpha"),
        "gravity": ("gamma_mantle", "alpha"),
    }
    EPS = 1e-6
    for setter, names in setters.items():
        for name in names:

            def _check_dependant_is_updated(setter, name):
                flex = Flexure1D(RasterModelGrid((3, 5)))
                val_before = 1. * getattr(flex, name)
                setattr(flex, setter, getattr(flex, setter) * (1. + EPS) + EPS)
                assert val_before != getattr(flex, name)

            _check_dependant_is_updated.description = "Test {0} updates {1}".format(
                setter, name
            )

            yield _check_dependant_is_updated, setter, name


def test_method_keyword():
    """Test using the method keyword."""
    flex = Flexure1D(RasterModelGrid((3, 5)), method="airy")
    assert flex.method == "airy"
    flex = Flexure1D(RasterModelGrid((3, 5)), method="flexure")
    assert flex.method == "flexure"
    with pytest.raises(ValueError):
        Flexure1D(RasterModelGrid((3, 5)), method="Flexure")


def test_constants_keywords():
    """Test the keywords for physical constants."""
    names = ("eet", "youngs", "rho_mantle", "rho_water", "gravity")
    for name in names:

        def _check_is_set(name):
            flex = Flexure1D(RasterModelGrid((3, 5)), **{name: 1.})
            assert getattr(flex, name) == 1.

        def _check_is_float(name):
            flex = Flexure1D(RasterModelGrid((3, 5)), **{name: 1})
            assert isinstance(getattr(flex, name), float)

        def _check_error_if_negative(name):
            with pytest.raises(ValueError):
                flex = Flexure1D(RasterModelGrid((3, 5)), **{name: -1})

        _check_is_set.description = "Test {name} keyword".format(name=name)
        _check_is_float.description = "Test {name} attribute is float".format(name=name)
        _check_error_if_negative.description = "Test {name} must not be negative".format(
            name=name
        )

        yield _check_is_set, name
        yield _check_is_float, name
        yield _check_error_if_negative, name


def test_x_at_node():
    """Test x_at_node is reshaped and shares memory with the grid."""
    flex = Flexure1D(RasterModelGrid((3, 5)))

    assert_array_equal(
        flex.x_at_node,
        [[0., 1., 2., 3., 4.], [0., 1., 2., 3., 4.], [0., 1., 2., 3., 4.]],
    )


def test_dz_at_node():
    """Test dz_at_node is reshaped and shares memory with its field."""
    flex = Flexure1D(RasterModelGrid((3, 5)))

    vals = flex.grid.at_node["lithosphere_surface__increment_of_elevation"]
    assert_array_equal(vals, 0.)

    assert np.may_share_memory(vals, flex.dz_at_node)
    assert flex.dz_at_node.shape == (3, 5)


def test_load_at_node():
    """Test load_at_node is reshaped and shares memory with its field."""
    flex = Flexure1D(RasterModelGrid((3, 5)))

    vals = flex.grid.at_node["lithosphere__increment_of_overlying_pressure"]
    assert_array_equal(vals, 0.)

    assert np.may_share_memory(vals, flex.load_at_node)
    assert flex.load_at_node.shape == (3, 5)


def test_x_is_contiguous():
    """Test that x_at_node is contiguous."""
    flex = Flexure1D(RasterModelGrid((3, 5)))
    assert flex.x_at_node.flags["C_CONTIGUOUS"]


def test_dz_is_contiguous():
    """Test that dz_at_node is contiguous."""
    flex = Flexure1D(RasterModelGrid((3, 5)))
    assert flex.dz_at_node.flags["C_CONTIGUOUS"]


def test_load_is_contiguous():
    """Test that load_at_node is contiguous."""
    flex = Flexure1D(RasterModelGrid((3, 5)))
    assert flex.load_at_node.flags["C_CONTIGUOUS"]
