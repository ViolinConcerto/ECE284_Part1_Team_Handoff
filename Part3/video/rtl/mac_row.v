module mac_row #(
    parameter bw = 4,
    parameter psum_bw = 32,
    parameter col = 8
)(
    input clk, reset, compute_clear,
    input [bw-1:0] in_w,
    input [psum_bw*col-1:0] in_n,
    output [psum_bw*col-1:0] out_s,
    input [1:0] inst_w,
    input act_signed,
    output [col-1:0] valid
);
    wire [(col+1)*bw-1:0] data_chain;
    wire [(col+1)*2-1:0] instruction_chain;
    wire [col:0] signed_chain;
    assign data_chain[bw-1:0] = in_w;
    assign instruction_chain[1:0] = inst_w;
    assign signed_chain[0] = act_signed;
    genvar c;
    generate for (c=0; c<col; c=c+1) begin : col_num
        mac_tile #(.bw(bw), .psum_bw(psum_bw)) mac_tile_instance(
            .clk(clk), .reset(reset), .compute_clear(compute_clear),
            .in_w(data_chain[c*bw +: bw]),
            .out_e(data_chain[(c+1)*bw +: bw]),
            .in_n(in_n[c*psum_bw +: psum_bw]),
            .out_s(out_s[c*psum_bw +: psum_bw]),
            .inst_w(instruction_chain[c*2 +: 2]),
            .inst_e(instruction_chain[(c+1)*2 +: 2]),
            .act_signed(signed_chain[c]), .act_signed_e(signed_chain[c+1]));
        assign valid[c] = instruction_chain[(c+1)*2+1];
    end endgenerate
endmodule
