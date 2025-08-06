// See README.md for license details.
package gemmini

import chisel3._
import chisel3.util._

class PEControl[T <: Data : Arithmetic](accType: T) extends Bundle {
  val dataflow = UInt(1.W) // TODO make this an Enum
  val propagate = UInt(1.W) // Which register should be propagated (and which should be accumulated)?
  val shift = UInt(log2Up(accType.getWidth).W) // TODO this isn't correct for Floats

}

// TODO update documentation
/**
  * A PE implementing a MAC operation. Configured as fully combinational when integrated into a Mesh.
  * @param width Data width of operands
  */
class PE[T <: Data](inputType: T, outputType: T, accType: T, df: Dataflow.Value, max_simultaneous_matmuls: Int, tile_row: UInt, tile_col: UInt, pe_row: UInt, pe_col: UInt)
                   (implicit ev: Arithmetic[T]) extends Module { // Debugging variables
  import ev._

  val io = IO(new Bundle {
    val in_a = Input(inputType)
    val in_b = Input(outputType)
    val in_d = Input(outputType)
    val out_a = Output(inputType)
    val out_b = Output(outputType)
    val out_c = Output(outputType)

    val in_control = Input(new PEControl(accType))
    val out_control = Output(new PEControl(accType))

    val in_id = Input(UInt(log2Up(max_simultaneous_matmuls).W))
    val out_id = Output(UInt(log2Up(max_simultaneous_matmuls).W))

    val in_last = Input(Bool())
    val out_last = Output(Bool())

    val in_valid = Input(Bool())
    val out_valid = Output(Bool())

    val bad_dataflow = Output(Bool())

    val fi_reg = Input(UInt(64.W))
    val fi_tile_col = Input(UInt(10.W))
    val fi_tile_row = Input(UInt(10.W))
    val fi_pe_row = Input(UInt(10.W))
    val fi_pe_col = Input(UInt(10.W))
    val do_fi = Input(UInt(1.W))
    val fault_model = Input(UInt(3.W))
    val fault_data = Input(UInt(20.W))
  })


  val fi_reg = Reg(UInt(64.W))
  fi_reg := fi_reg

  val fi_tile_col = io.fi_tile_col
  val fi_tile_row = io.fi_tile_row
  val fi_pe_col = io.fi_pe_col
  val fi_pe_row = io.fi_pe_row
  val do_fi = io.do_fi
  
  val fault_model = io.fault_model
  val fault_data = io.fault_data

  val cType = if (df == Dataflow.WS) inputType else accType

  val a  = io.in_a
  val b  = io.in_b
  val d  = io.in_d
  val c1 = Reg(cType)
  val c2 = Reg(cType)
  val dataflow = io.in_control.dataflow
  val prop  = io.in_control.propagate
  val shift = io.in_control.shift
  val id = io.in_id
  val last = io.in_last
  val valid = io.in_valid

  io.out_a := a
  io.out_control.dataflow := dataflow
  io.out_control.propagate := prop
  io.out_control.shift := shift
  io.out_id := id
  io.out_last := last
  io.out_valid := valid

  val last_s = RegEnable(prop, valid)
  val flip = last_s =/= prop
  val shift_offset = Mux(flip, shift, 0.U)

  // Which dataflow are we using?
  val OUTPUT_STATIONARY = Dataflow.OS.id.U(1.W)
  val WEIGHT_STATIONARY = Dataflow.WS.id.U(1.W)

  // Is c1 being computed on, or propagated forward (in the output-stationary dataflow)?
  val COMPUTE = 0.U(1.W)
  val PROPAGATE = 1.U(1.W)

  def macWithFIMult(a: T, b: T, acc: T): T = {
    val mult = a * b
    val faulty = mult.injectFault(tile_row, tile_col, pe_row, pe_col,
                                  do_fi, fi_tile_row, fi_tile_col,
                                  fi_pe_row, fi_pe_col, fault_model, fault_data)
    return acc + faulty
  }

  def macWithFI(a: T, b: T, acc: T): T = {
    val faulty = acc + a * b
    return faulty.injectFault(tile_row, tile_col, pe_row, pe_col,
                                  do_fi, fi_tile_row, fi_tile_col,
                                  fi_pe_row, fi_pe_col, fault_model, fault_data)
  }

  val inject_here = do_fi === 1.U &&
                  tile_row === fi_tile_row && tile_col === fi_tile_col &&
                  pe_row === fi_pe_row && pe_col === fi_pe_col &&
                  a.isNonZero(a) && b.isNonZero(b)

  io.bad_dataflow := false.B
  when ((df == Dataflow.OS).B || ((df == Dataflow.BOTH).B && dataflow === OUTPUT_STATIONARY)) {

    assert(fi_tile_col < 1024.U && fi_tile_row < 1024.U && fi_pe_row < 1024.U && fi_pe_col < 1024.U && do_fi < 2.U && fault_model < 5.U)
    
    when(prop === PROPAGATE) {
      io.out_c := (c1 >> shift_offset).clippedToWidthOf(outputType)
      io.out_b := b
      
      when(inject_here) {
        c2 := macWithFIMult(a, b.asTypeOf(inputType), c2)
      }.otherwise {
        c2 := c2.mac(a, b.asTypeOf(inputType))
      }
      
      c1 := d.withWidthOf(cType)
    }.otherwise {
      io.out_c := (c2 >> shift_offset).clippedToWidthOf(outputType)
      io.out_b := b
      
      when(inject_here) {
        c1 := macWithFIMult(a, b.asTypeOf(inputType), c1)
      }.otherwise {
        c1 := c1.mac(a, b.asTypeOf(inputType))
      }

      c2 := d.withWidthOf(cType)
    }
  }.elsewhen ((df == Dataflow.WS).B || ((df == Dataflow.BOTH).B && dataflow === WEIGHT_STATIONARY)) {
    when(prop === PROPAGATE) {
      io.out_c := c1

      when(inject_here) {
        io.out_b := macWithFIMult(a, c2.asTypeOf(inputType), b)
      }.otherwise {
        io.out_b := b.mac(a, c2.asTypeOf(inputType))
      }
      
      c1 := d
    }.otherwise {
      io.out_c := c2

      when(inject_here) {
        io.out_b := macWithFIMult(a, c1.asTypeOf(inputType), b)
      }.otherwise {
        io.out_b := b.mac(a, c1.asTypeOf(inputType))
      }
      
      c2 := d
    }
  }.otherwise {
    io.bad_dataflow := true.B
    //assert(false.B, "unknown dataflow")
    io.out_c := DontCare
    io.out_b := DontCare
  }

  when (!valid) {
    c1 := c1
    c2 := c2
  }
}