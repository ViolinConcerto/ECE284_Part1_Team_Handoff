module corelet #(
    parameter row = 8,
    parameter col = 8,
    parameter bw = 4,
    parameter psum_bw = 32
)(
    input clk, reset, compute_clear,
    input [33:0] inst,
    input act_signed,
    input sfp_clear, sfp_relu, pmem_from_sfp,
    input [4:0] sfp_shift,
    input [row*bw-1:0] l0_in,
    input [col*psum_bw-1:0] sfp_in,
    output l0_o_full, l0_o_ready, ofifo_o_full, ofifo_o_ready, ofifo_valid,
    output [col*psum_bw-1:0] ofifo_out, sfp_out, pmem_write_data
);
    wire local_clear = reset || compute_clear;
    wire [row*bw-1:0] mac_in_w;
    wire [col*psum_bw-1:0] mac_out_s;
    wire [col-1:0] mac_valid;
    assign pmem_write_data = pmem_from_sfp ? sfp_out : ofifo_out;
    l0 #(.row(row), .bw(bw)) l0_instance(
        .clk(clk), .reset(local_clear), .wr(inst[2]), .rd(inst[3]),
        .in(l0_in), .out(mac_in_w), .o_full(l0_o_full), .o_ready(l0_o_ready));
    mac_array #(.bw(bw), .psum_bw(psum_bw), .row(row), .col(col)) mac_array_instance(
        .clk(clk), .reset(reset), .compute_clear(compute_clear),
        .in_w(mac_in_w), .inst_w(inst[1:0]), .act_signed(act_signed),
        .out_s(mac_out_s), .valid(mac_valid));
    ofifo #(.col(col), .bw(psum_bw)) ofifo_instance(
        .clk(clk), .reset(local_clear), .wr(mac_valid), .rd(inst[6]),
        .in(mac_out_s), .out(ofifo_out), .o_full(ofifo_o_full),
        .o_ready(ofifo_o_ready), .o_valid(ofifo_valid));
    genvar c;
    generate for (c=0; c<col; c=c+1) begin : sfp_num
        sfp #(.bw(bw), .psum_bw(psum_bw)) sfp_instance(
            .clk(clk), .reset(local_clear), .clear(sfp_clear), .acc(inst[33]),
            .relu(sfp_relu), .shift(sfp_shift),
            .in(sfp_in[c*psum_bw +: psum_bw]), .out(sfp_out[c*psum_bw +: psum_bw]));
    end endgenerate
endmodule
