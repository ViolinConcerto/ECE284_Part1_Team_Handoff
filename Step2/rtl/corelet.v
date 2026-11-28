// Step2 datapath; SRAM instances belong to core, not this FPGA top.
module corelet #(
    parameter row = 8,
    parameter col = 8,
    parameter bw = 4,
    parameter psum_bw = 16,
    parameter STREAM_OFIFO = 0,
    parameter OFIFO_DEPTH = 16
)(
    input clk,
    input reset,
    input [33:0] inst,
    input [row*bw-1:0] l0_in,
    input [col*psum_bw-1:0] sfp_in,
    input sfp_clear,
    input sfp_relu,
    input pmem_from_sfp,
    output l0_o_full,
    output l0_o_ready,
    output ofifo_o_full,
    output ofifo_o_ready,
    output ofifo_valid,
    output [col*psum_bw-1:0] ofifo_out,
    output [col*psum_bw-1:0] sfp_out,
    output [col*psum_bw-1:0] pmem_write_data
);
    wire [row*bw-1:0] west_data;
    wire [col*psum_bw-1:0] array_result;
    wire [col-1:0] array_valid;

    l0 #(.row(row), .bw(bw)) input_buffer(
        .clk(clk), .reset(reset), .in(l0_in), .out(west_data),
        .rd(inst[3]), .wr(inst[2]), .o_full(l0_o_full), .o_ready(l0_o_ready));

    mac_array #(.row(row), .col(col), .bw(bw), .psum_bw(psum_bw)) processing_array(
        .clk(clk), .reset(reset), .in_w(west_data),
        .in_n({col*psum_bw{1'b0}}), .inst_w(inst[1:0]),
        .out_s(array_result), .valid(array_valid));

    generate if (STREAM_OFIFO) begin : stream_output
        ofifo_stream #(.COLS(col), .WIDTH(psum_bw), .DEPTH(OFIFO_DEPTH)) output_buffer(
            .clk(clk), .reset(reset), .in(array_result), .out(ofifo_out),
            .wr(array_valid), .rd(inst[6]), .o_full(ofifo_o_full), .o_valid(ofifo_valid));
        assign ofifo_o_ready = !ofifo_o_full;
    end else begin : buffered_output
        ofifo #(.col(col), .bw(psum_bw)) output_buffer(
            .clk(clk), .reset(reset), .in(array_result), .out(ofifo_out),
            .wr(array_valid), .rd(inst[6]), .o_full(ofifo_o_full),
            .o_ready(ofifo_o_ready), .o_valid(ofifo_valid));
    end endgenerate

    genvar channel;
    generate for (channel=0; channel<col; channel=channel+1) begin : output_channel
        sfp #(.bw(bw), .psum_bw(psum_bw)) accumulator(
            .clk(clk), .reset(reset), .clear(sfp_clear), .acc(inst[33]), .relu(sfp_relu),
            .in(sfp_in[channel*psum_bw +: psum_bw]),
            .out(sfp_out[channel*psum_bw +: psum_bw]));
    end endgenerate

    assign pmem_write_data = pmem_from_sfp ? sfp_out : ofifo_out;
endmodule
