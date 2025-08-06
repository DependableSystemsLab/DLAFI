// See README.md for license details.

package gemmini

import chisel3._
import chisel3.util._
import Util._

/**
  * A Tile is a purely combinational 2D array of passThrough PEs.
  * a, b, s, and in_propag are broadcast across the entire array and are passed through to the Tile's outputs
  * @param width The data width of each PE in bits
  * @param rows Number of PEs on each row
  * @param columns Number of PEs on each column
  */
class Tile[T <: Data](inputType: T, outputType: T, accType: T, df: Dataflow.Value, tree_reduction: Boolean, max_simultaneous_matmuls: Int, val rows: Int, val columns: Int, tile_row: UInt, tile_col: UInt)(implicit ev: Arithmetic[T]) extends Module {
  val io = IO(new Bundle {
    val in_a        = Input(Vec(rows, inputType))
    val in_b        = Input(Vec(columns, outputType)) // This is the output of the tile next to it
    val in_d        = Input(Vec(columns, outputType))

    val in_control  = Input(Vec(columns, new PEControl(accType)))
    val in_id       = Input(Vec(columns, UInt(log2Up(max_simultaneous_matmuls).W)))
    val in_last  = Input(Vec(columns, Bool()))

    val out_a       = Output(Vec(rows, inputType))
    val out_c       = Output(Vec(columns, outputType))
    val out_b       = Output(Vec(columns, outputType))

    val out_control = Output(Vec(columns, new PEControl(accType)))
    val out_id      = Output(Vec(columns, UInt(log2Up(max_simultaneous_matmuls).W)))
    val out_last    = Output(Vec(columns, Bool()))

    val in_valid = Input(Vec(columns, Bool()))
    val out_valid = Output(Vec(columns, Bool()))

    val bad_dataflow = Output(Bool())

    val fi_reg = Input(UInt(64.W))
  })

  import ev._

  val fi_reg = Reg(UInt(64.W))
  fi_reg := io.fi_reg

  val tile = Seq.tabulate(rows, columns)( (i, j) => Module(new PE(inputType, outputType, accType, df, max_simultaneous_matmuls, tile_row, tile_col, i.U, j.U)))
  val tileT = tile.transpose

  for (r <- 0 until rows) {
    for (c <- 0 until columns) {
      tile(r)(c).io.fi_reg := fi_reg
      tile(r)(c).io.fi_tile_col := fi_reg(9, 0)
      tile(r)(c).fi_tile_row := fi_reg(19, 10)
      tile(r)(c).fi_pe_row := fi_reg(29, 20)
      tile(r)(c).fi_pe_col := fi_reg(39, 30)
      tile(r)(c).do_fi := fi_reg(40)
      tile(r)(c).fault_model := fi_reg(43, 41)
      tile(r)(c).fault_data := fi_reg(63, 44)
    }
  }

  // TODO: abstract hori/vert broadcast, all these connections look the same
  // Broadcast 'a' horizontally across the Tile
  for (r <- 0 until rows) {
    tile(r).foldLeft(io.in_a(r)) {
      case (in_a, pe) =>
        pe.io.in_a := in_a
        pe.io.out_a
    }
  }

  // Broadcast 'b' vertically across the Tile
  for (c <- 0 until columns) {
    tileT(c).foldLeft(io.in_b(c)) {
      case (in_b, pe) =>
        pe.io.in_b := (if (tree_reduction) in_b.zero else in_b)
        pe.io.out_b
    }
  }

  // Broadcast 'd' vertically across the Tile
  for (c <- 0 until columns) {
    tileT(c).foldLeft(io.in_d(c)) {
      case (in_d, pe) =>
        pe.io.in_d := in_d
        pe.io.out_c
    }
  }

  // Broadcast 'control' vertically across the Tile
  for (c <- 0 until columns) {
    tileT(c).foldLeft(io.in_control(c)) {
      case (in_ctrl, pe) =>
        pe.io.in_control := in_ctrl
        pe.io.out_control
    }
  }

  // Broadcast 'garbage' vertically across the Tile
  for (c <- 0 until columns) {
    tileT(c).foldLeft(io.in_valid(c)) {
      case (v, pe) =>
        pe.io.in_valid := v
        pe.io.out_valid
    }
  }

  // Broadcast 'id' vertically across the Tile
  for (c <- 0 until columns) {
    tileT(c).foldLeft(io.in_id(c)) {
      case (id, pe) =>
        pe.io.in_id := id
        pe.io.out_id
    }
  }

  // Broadcast 'last' vertically across the Tile
  for (c <- 0 until columns) {
    tileT(c).foldLeft(io.in_last(c)) {
      case (last, pe) =>
        pe.io.in_last := last
        pe.io.out_last
    }
  }

  // Drive the Tile's bottom IO
  for (c <- 0 until columns) {
    io.out_c(c) := tile(rows-1)(c).io.out_c
    io.out_control(c) := tile(rows-1)(c).io.out_control
    io.out_id(c) := tile(rows-1)(c).io.out_id
    io.out_last(c) := tile(rows-1)(c).io.out_last
    io.out_valid(c) := tile(rows-1)(c).io.out_valid

    io.out_b(c) := {
      if (tree_reduction) {
        val prods = tileT(c).map(_.io.out_b)
        accumulateTree(prods :+ io.in_b(c))
      } else {
        tile(rows - 1)(c).io.out_b
      }
    }
  }
  io.bad_dataflow := tile.map(_.map(_.io.bad_dataflow).reduce(_||_)).reduce(_||_)

  // Drive the Tile's right IO
  for (r <- 0 until rows) {
    io.out_a(r) := tile(r)(columns-1).io.out_a
  }
}