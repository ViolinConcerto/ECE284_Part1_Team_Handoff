// Step2 core with the Step3 testbench's control and observation ports.
module core #(
    parameter bw = 4,
    parameter psum_bw = 16,
    parameter col = 8,
    parameter row = 8,
    parameter STREAM_OFIFO = 0,
    parameter OFIFO_DEPTH = 16
)(
    input clk,
    input reset,
    input [33:0] inst,
    input [31:0] D_xmem,
    input sfp_clear,
    input sfp_relu,
    input pmem_from_sfp,
    output ofifo_valid,
    output [127:0] sfp_out,
    output [127:0] pmem_read_data,
    output [31:0] xmem_read_data,
    output l0_ready,
    output ofifo_full
);
    wire [31:0] input_word;
    wire [127:0] psum_word;
    wire [127:0] psum_write_word;

    sram_32b_w2048 input_memory(
        .CLK(clk), .D(D_xmem), .Q(input_word),
        .CEN(inst[19]), .WEN(inst[18]), .A(inst[17:7]));

    sram_128b_w2048 partial_sum_memory(
        .CLK(clk), .D(psum_write_word), .Q(psum_word),
        .CEN(inst[32]), .WEN(inst[31]), .A(inst[30:20]));

    corelet #(.row(row), .col(col), .bw(bw), .psum_bw(psum_bw),
              .STREAM_OFIFO(STREAM_OFIFO), .OFIFO_DEPTH(OFIFO_DEPTH)) datapath(
        .clk(clk), .reset(reset), .inst(inst), .l0_in(input_word), .sfp_in(psum_word),
        .sfp_clear(sfp_clear), .sfp_relu(sfp_relu), .pmem_from_sfp(pmem_from_sfp),
        .l0_o_full(), .l0_o_ready(l0_ready), .ofifo_o_full(ofifo_full), .ofifo_o_ready(),
        .ofifo_valid(ofifo_valid), .ofifo_out(), .sfp_out(sfp_out),
        .pmem_write_data(psum_write_word));

    assign pmem_read_data = psum_word;
    assign xmem_read_data = input_word; // lets the controller read the occupancy bitmap
endmodule
